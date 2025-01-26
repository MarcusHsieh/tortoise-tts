from micinput import MicRecorder
from whisperSTT import transcribe_audio
from nlpkermit import KermitResponder

import tkinter as tk
from threading import Thread
from PIL import Image, ImageTk

import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torchaudio
import wave
import math
import datetime
from typing import List

# pydub for playback & pitch shift
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

# Tortoise imports
from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices

class KermitGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Kerminator")
        self.root.geometry("1280x800")
        self.root.configure(bg="#C0D3C5")

        # init kermit
        self.kermit = KermitResponder()
        self.mic_recorder = MicRecorder(file_name="micinput.wav")

        self.is_recording = False
        self.is_processing = False

        # The folder for the *current conversation*
        self.conversation_folder = None

        self._create_ui()
        self.root.bind("<space>", self.toggle_microphone)

    def _create_ui(self):
        try:
            from PIL import Image
            kermit_image = Image.open("public/kermit-think.png")
            kermit_image = kermit_image.resize((200, 200), Image.LANCZOS)
            self.kermit_photo = ImageTk.PhotoImage(kermit_image)
            kermit_image_label = tk.Label(self.root, image=self.kermit_photo, bg="#C0D3C5")
            kermit_image_label.place(relx=0.5, rely=0.17, anchor="center")
        except:
            print("Error: 'kermit-think.png' not found or PIL not installed.")

        frame_width = 700
        frame_height = 500
        frame = tk.Frame(self.root, bg="#466362", width=frame_width, height=frame_height, relief="ridge", bd=2)
        frame.place(relx=0.5, rely=0.6, anchor="center")
        frame.pack_propagate(False)

        tk.Label(frame, text=" User Input: ", bg="#466362", font=("Helvetica", 14), height=2).pack(pady=10)
        user_input_frame = tk.Frame(frame, bg="#466362")
        user_input_frame.pack(pady=5, fill="both", expand=True)

        self.user_input_text = tk.Text(
            user_input_frame, wrap="word", bg="#e8e8e8", fg="#333333",
            font=("Helvetica", 12), relief="solid", height=5, width=60,
            padx=5, pady=5, spacing3=3
        )
        self.user_input_text.pack(pady=5)

        tk.Label(frame, text=" Kermit Response: ", bg="#466362", font=("Helvetica", 14), height=2).pack(pady=10)
        kermit_response_frame = tk.Frame(frame, bg="#466362")
        kermit_response_frame.pack(pady=5, fill="both", expand=True)

        self.kermit_response_text = tk.Text(
            kermit_response_frame, wrap="word", bg="#e8e8e8", fg="#333333",
            font=("Helvetica", 12), relief="solid", height=8, width=60,
            padx=5, pady=5, spacing3=3
        )
        self.kermit_response_text.pack(pady=5)

        self.mic_button = tk.Button(
            frame, text="Press Spacebar to Record", command=self.toggle_microphone,
            bg="#4caf50", fg="white", font=("Helvetica", 14), relief="flat",
            activebackground="#388e3c", activeforeground="white", height=2, width=30
        )
        self.mic_button.pack(pady=20)

    def toggle_microphone(self, event=None):
        if self.is_processing:
            self.update_text(self.kermit_response_text, "Pipeline busy. Please wait.")
            return

        if self.is_recording:
            self.stop_recording()
            Thread(target=self.process_audio, daemon=True).start()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.mic_button.config(text="Recording... (Space to stop)",
                               bg="#d32f2f", activebackground="#c62828")
        self.clear_text(self.user_input_text)
        self.clear_text(self.kermit_response_text)
        self.mic_recorder.start_recording()

        # If no conversation folder yet, create one now
        if self.conversation_folder is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.conversation_folder = os.path.join("results", f"conversation_{timestamp}")
            os.makedirs(self.conversation_folder, exist_ok=True)

    def stop_recording(self):
        self.is_recording = False
        self.mic_button.config(text="Press Spacebar to Record",
                               bg="#4caf50", activebackground="#388e3c")
        self.mic_recorder.stop_recording()

    def process_audio(self):
        """
        1) Transcribe
        2) Kermit NLP
        3) TTS
        4) Save WAV
        5) Pitch-shift, re-save
        6) Playback
        7) Log amplitude/time
        """
        try:
            self.is_processing = True

            # 1) Transcribe
            self.update_text(self.user_input_text, "Transcribing audio...")
            transcribed_text = transcribe_audio(file_name="micinput.wav")
            self.update_text(self.user_input_text, transcribed_text)

            # 2) Kermit response
            self.update_text(self.kermit_response_text, "Generating response...")
            response = self.kermit.get_response(transcribed_text)
            self.update_text(self.kermit_response_text, response)

            # 3) Tortoise TTS
            # self.update_text(self.user_input_text, "Synthesizing with Tortoise TTS...")

            tts = TextToSpeech(models_dir=MODELS_DIR, use_deepspeed=False, kv_cache=True, half=True)
            voice_samples, conditioning_latents = load_voices(["newkermit"])

            gen, dbg_state = tts.tts_with_preset(
                response,
                k=1,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                preset="ultra_fast",
                use_deterministic_seed=None,
                return_deterministic_state=True,
                cvvp_amount=0.0
            )

            if isinstance(gen, list):
                gen_audio = gen[0]
            else:
                gen_audio = gen

            # 4) Save the WAV -> conversation folder
            wav_count = len([f for f in os.listdir(self.conversation_folder) if f.endswith(".wav")])
            base_wav_filename = f"newkermit_{wav_count+1:03d}.wav"
            outpath = os.path.join(self.conversation_folder, base_wav_filename)

            torchaudio.save(outpath, gen_audio.squeeze(0).cpu(), 24000)

            # 5) Pitch shift using pydub (optional)
            # e.g. +3 semitones for a higher pitch
            # Then re-export to new file: e.g. "newkermit_001_pitched.wav"
            pitched_filename = base_wav_filename.replace(".wav", "_pitched.wav")
            pitched_outpath = os.path.join(self.conversation_folder, pitched_filename)

            if HAS_PYDUB:
                # self.update_text(self.kermit_response_text, f"Applying pitch shift to {base_wav_filename} ...")
                seg = AudioSegment.from_wav(outpath)
                seg_pitched = self.pitch_shift_pydub(seg, semitones=1.5)
                seg_pitched.export(pitched_outpath, format="wav")
                # self.update_text(self.kermit_response_text, f"Playing pitched audio {pitched_filename} ...")
                pydub_play(seg_pitched)

                # amplitude logging on the pitched file
                amplitude_log = self.compute_rms_over_time(pitched_outpath)
                log_file = os.path.join(self.conversation_folder, "amplitude_log.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n=== WAV: {pitched_filename} (pitch +1.5 semitones) ===\n")
                    for (t_sec, amp) in amplitude_log:
                        f.write(f"{t_sec:.2f}s => {amp:.4f}\n")

            else:
                # If pydub not installed, just play the normal file
                # self.update_text(self.kermit_response_text, f"Pydub not installed. Playing {base_wav_filename} without pitch shift.")
                amplitude_log = self.compute_rms_over_time(outpath)
                log_file = os.path.join(self.conversation_folder, "amplitude_log.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n=== WAV: {base_wav_filename} (no pitch shift) ===\n")
                    for (t_sec, amp) in amplitude_log:
                        f.write(f"{t_sec:.2f}s => {amp:.4f}\n")

        except Exception as e:
            self.update_text(self.kermit_response_text, f"Error: {str(e)}")
        finally:
            self.is_processing = False

    def pitch_shift_pydub(self, seg: AudioSegment, semitones: float = 2.0) -> AudioSegment:
        """
        Increase/decrease pitch by a certain number of semitones using pydub.
        This approach changes the frame_rate, then resets it, so the audio
        length remains about the same, but the pitch is shifted.
        """
        # semitone ratio => 2^(n/12). For +3 semitones => factor ~1.189
        factor = 2.0 ** (semitones / 12.0)
        new_frame_rate = int(seg.frame_rate * factor)

        # speed up the playback rate
        seg_pitched = seg._spawn(seg.raw_data, overrides={"frame_rate": new_frame_rate})
        # Then set it back to the original frame rate => pitch shift without speed change
        seg_pitched = seg_pitched.set_frame_rate(seg.frame_rate)
        return seg_pitched

    def compute_rms_over_time(self, wav_path: str, chunk_ms=50):
        seg = AudioSegment.from_wav(wav_path)
        amplitude_data = []
        current_ms = 0
        while current_ms < len(seg):
            chunk = seg[current_ms:current_ms+chunk_ms]
            rms = chunk.rms
            amplitude = rms / 32767.0
            amplitude_data.append((current_ms / 1000.0, amplitude))
            current_ms += chunk_ms
        return amplitude_data

    def update_text(self, text_widget, content):
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, content)
        text_widget.see(tk.END)

    def clear_text(self, text_widget):
        text_widget.delete(1.0, tk.END)

    def run(self):
        self.root.mainloop()
        self.mic_recorder.terminate()


if __name__ == "__main__":
    app = KermitGUI()
    app.run()

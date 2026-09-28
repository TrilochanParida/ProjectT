import subprocess
import os
from faster_whisper import WhisperModel

RAW_AUDIO = "voice/live_raw.wav"
MONO_AUDIO = "voice/live_mono.wav"

print("\n========== PROJECT T ==========")
print("Speak your sentence after the recording starts.")
print("Recording for 5 seconds...\n")

# Record from INMP441
subprocess.run([
    "arecord",
    "-D", "plughw:1,0",
    "-f", "S32_LE",
    "-r", "48000",
    "-c", "2",
    "-d", "5",
    RAW_AUDIO
], check=True)

print("\nRecording complete.")
print("Converting audio...")

# Keep left I2S channel and convert to 16 kHz mono
subprocess.run([
    "ffmpeg",
    "-y",
    "-i", RAW_AUDIO,
    "-af", "pan=mono|c0=c0",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    MONO_AUDIO
], check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

print("Loading Whisper...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("Transcribing...\n")

segments, info = model.transcribe(
    MONO_AUDIO,
    language="en",
    beam_size=1
)

text = ""

for segment in segments:
    text += segment.text

print("========== YOU SAID ==========")
print(text.strip())
print("==============================\n")

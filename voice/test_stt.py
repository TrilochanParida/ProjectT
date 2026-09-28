from faster_whisper import WhisperModel

AUDIO_FILE = "voice/voice_mono.wav"

print("Loading Whisper model...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("Transcribing...")

segments, info = model.transcribe(
    AUDIO_FILE,
    language="en",
    beam_size=1
)

print("\n========== PROJECT T STT ==========")

text = ""

for segment in segments:
    text += segment.text

print(text.strip())
print("===================================")

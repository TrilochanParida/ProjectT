import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from faster_whisper import WhisperModel
from google import genai
from config.secrets import GEMINI_API_KEY


RAW_AUDIO = "voice/live_raw.wav"
MONO_AUDIO = "voice/live_mono.wav"


def record_audio():
    print("\n🎤 Listening for 5 seconds...")

    subprocess.run([
        "arecord",
        "-D", "plughw:1,0",
        "-f", "S32_LE",
        "-r", "48000",
        "-c", "2",
        "-d", "5",
        RAW_AUDIO
    ], check=True)


def convert_audio():
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


def transcribe(model):
    print("🧠 Converting speech to text...")

    segments, info = model.transcribe(
        MONO_AUDIO,
        language="en",
        beam_size=1
    )

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()


def ask_gemini(client, user_text):
    print("🤖 Asking Gemini...")

    prompt = f"""
You are Project T, an intelligent companion robot.

The user said:
"{user_text}"

Reply naturally and briefly.
Keep your answer suitable for being spoken aloud by a small robot.
Do not use markdown.
"""

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash"
    ]

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            print(f"{model} unavailable, trying next model...")

    return "Sorry, I am having trouble connecting to my AI brain."


def main():

    print("\n================================")
    print("       PROJECT T AI TEST")
    print("================================")

    print("\nLoading Whisper...")
    whisper = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )

    client = genai.Client(api_key=GEMINI_API_KEY)

    # 1. Record
    record_audio()

    # 2. Convert
    print("Processing audio...")
    convert_audio()

    # 3. Speech → text
    user_text = transcribe(whisper)

    print("\n========== YOU SAID ==========")
    print(user_text)
    print("==============================")

    if not user_text:
        print("No speech detected.")
        return

    # 4. Text → Gemini
    answer = ask_gemini(client, user_text)

    print("\n========== PROJECT T ==========")
    print(answer)
    print("===============================\n")


if __name__ == "__main__":
    main()

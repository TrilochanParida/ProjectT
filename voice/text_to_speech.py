import asyncio
import subprocess
import edge_tts

OUTPUT_MP3 = "voice/response.mp3"
OUTPUT_WAV = "voice/response.wav"


async def generate_speech(text):
    communicate = edge_tts.Communicate(
        text,
        "en-IN-NeerjaNeural"
    )

    await communicate.save(OUTPUT_MP3)


def speak(text):
    print("🔊 Generating speech...")

    asyncio.run(generate_speech(text))

    print("🔊 Converting audio...")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", OUTPUT_MP3,
        "-ar", "48000",
        "-ac", "2",
        "-sample_fmt", "s16",
        OUTPUT_WAV
    ], check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("🔊 Project T speaking...")

    subprocess.run([
        "aplay",
        "-D", "plughw:1,1",
        OUTPUT_WAV
    ], check=True)

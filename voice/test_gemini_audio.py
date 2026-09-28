import asyncio
import os
import sys
import wave
import subprocess

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from google import genai
from google.genai import types
from config.secrets import GEMINI_API_KEY


MODEL = "gemini-3.8-live"

MIC = "plughw:1,0"
INPUT_RATE = 16000
CHUNK_SIZE = 1280


def start_microphone():
    return subprocess.Popen(
        [
            "arecord",
            "-D", MIC,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-t", "raw",
            "-q",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )


async def microphone_loop(session, mic):
    while True:
        data = await asyncio.to_thread(
            mic.stdout.read,
            CHUNK_SIZE
        )

        if not data:
            break

        await session.send_realtime_input(
            audio=types.Blob(
                data=data,
                mime_type="audio/pcm;rate=16000"
            )
        )


async def receive_loop(session):
    audio_data = bytearray()

    async for response in session.receive():

        if not response.server_content:
            continue

        content = response.server_content

        if content.input_transcription:
            text = content.input_transcription.text
            if text:
                print("YOU:", text)

        if content.model_turn:

            for part in content.model_turn.parts:

                if part.inline_data:

                    data = part.inline_data.data

                    print(
                        "GEMINI AUDIO CHUNK:",
                        len(data),
                        "bytes"
                    )

                    audio_data.extend(data)

                if part.text:
                    print("TEXT:", part.text)

        if content.turn_complete:

            print()
            print("TURN COMPLETE")
            print("Total audio:", len(audio_data), "bytes")

            if audio_data:

                filename = "voice/gemini_response.raw"

                with open(filename, "wb") as f:
                    f.write(audio_data)

                print("Saved:", filename)

                wav_name = "voice/gemini_response.wav"

                with wave.open(wav_name, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(audio_data)

                print("Saved:", wav_name)

            print()
            return


async def main():

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    mic = start_microphone()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],

        system_instruction=types.Content(
            parts=[
                types.Part(
                    text="""
                    You are Zeno, a friendly AI companion robot.
                    Answer briefly and naturally.
                    """
                )
            ]
        ),

        input_audio_transcription=
            types.AudioTranscriptionConfig(),

        output_audio_transcription=
            types.AudioTranscriptionConfig(),

        realtime_input_config=
            types.RealtimeInputConfig(
                automatic_activity_detection=
                    types.AutomaticActivityDetection(
                        disabled=False,
                        prefix_padding_ms=100,
                        silence_duration_ms=600
                    )
            )
    )

    try:

        async with client.aio.live.connect(
            model=MODEL,
            config=config
        ) as session:

            print("🟢 GEMINI CONNECTED")
            print("🎤 Speak one sentence...")
            print()

            mic_task = asyncio.create_task(
                microphone_loop(session, mic)
            )

            receive_task = asyncio.create_task(
                receive_loop(session)
            )

            await receive_task

            mic_task.cancel()

            await asyncio.gather(
                mic_task,
                return_exceptions=True
            )

    finally:

        mic.terminate()

        try:
            mic.wait(timeout=1)
        except:
            mic.kill()


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import subprocess
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from google import genai
from google.genai import types
from config.secrets import GEMINI_API_KEY


MODEL = "gemini-3.1-flash-live-preview"

MIC = "plughw:1,0"
SPEAKER = "plughw:1,1"

CHUNK_SIZE = 1280


SYSTEM = """
You are Zeno, the AI voice assistant inside Project T,
a self-balancing intelligent companion robot.

Speak naturally, warmly and briefly.

You are having a continuous voice conversation with the user.

Answer questions directly.

You can speak in English, Hindi, or Hinglish depending
on how the user speaks.

Remember the conversation context.

Do not unnecessarily repeat yourself.
"""


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


def play_audio(audio):

    if not audio:
        return

    speaker = subprocess.Popen(
        [
            "aplay",
            "-D", SPEAKER,
            "-f", "S16_LE",
            "-r", "24000",
            "-c", "1",
            "-t", "raw",
            "-"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:

        speaker.stdin.write(audio)
        speaker.stdin.close()
        speaker.wait()

    except Exception as e:

        print("Speaker error:", e)

        try:
            speaker.kill()
        except Exception:
            pass


async def microphone_loop(session, mic, listening):

    print("🎤 Microphone streaming...")

    while True:

        data = await asyncio.to_thread(
            mic.stdout.read,
            CHUNK_SIZE
        )

        if not data:
            print("⚠️ Microphone stopped.")
            return

        # Only send microphone audio when Zeno is listening.
        #
        # During playback we still read the microphone so the
        # arecord buffer does not fill, but we discard the audio.

        if listening.is_set():

            await session.send_realtime_input(
                audio=types.Blob(
                    data=data,
                    mime_type="audio/pcm;rate=16000"
                )
            )


async def receive_loop(session, listening):

    print("🔊 Gemini receiver running...")
    print()

    audio_buffer = bytearray()

    async for response in session.receive():

        if not response.server_content:
            continue

        content = response.server_content

        # -----------------------------
        # USER TRANSCRIPTION
        # -----------------------------

        if content.input_transcription:

            text = content.input_transcription.text

            if text:

                print(
                    f"\n👤 YOU: {text}",
                    flush=True
                )

        # -----------------------------
        # GEMINI RESPONSE
        # -----------------------------

        if content.model_turn:

            for part in content.model_turn.parts:

                if part.inline_data:

                    audio_buffer.extend(
                        part.inline_data.data
                    )

                if part.text:

                    print(
                        f"🤖 ZENO: {part.text}",
                        end="",
                        flush=True
                    )

        # -----------------------------
        # RESPONSE FINISHED
        # -----------------------------

        if content.turn_complete:

            print()

            if audio_buffer:

                audio = bytes(audio_buffer)

                print(
                    f"🔊 Playing {len(audio)} bytes..."
                )

                # Stop sending microphone audio BEFORE
                # playing Zeno's response.
                listening.clear()

                await asyncio.to_thread(
                    play_audio,
                    audio
                )

                # Now Zeno has finished speaking.
                # Start listening again.
                listening.set()

                print("✅ Playback complete.")
                print("🎤 Listening for next question...")

            audio_buffer.clear()

            print()


async def main():

    print()
    print("======================================")
    print("       PROJECT T + ZENO LIVE")
    print("======================================")
    print()
    print(f"Model: {MODEL}")
    print()

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    mic = start_microphone()

    # True = microphone audio is sent to Gemini.
    # False = microphone is read but discarded.
    listening = asyncio.Event()
    listening.set()

    config = types.LiveConnectConfig(

        response_modalities=["AUDIO"],

        system_instruction=types.Content(
            parts=[
                types.Part(
                    text=SYSTEM
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

            print("🟢 GEMINI LIVE CONNECTED")
            print("--------------------------------------")
            print("Zeno is continuously listening.")
            print("Speak naturally.")
            print("Wait until Zeno finishes speaking.")
            print("Then ask your next question.")
            print("Press Ctrl+C to stop.")
            print("--------------------------------------")
            print()

            mic_task = asyncio.create_task(
                microphone_loop(
                    session,
                    mic,
                    listening
                )
            )

            receive_task = asyncio.create_task(
                receive_loop(
                    session,
                    listening
                )
            )

            await asyncio.gather(
                mic_task,
                receive_task
            )

    finally:

        print()
        print("Stopping microphone...")

        try:

            mic.terminate()
            mic.wait(timeout=1)

        except Exception:

            try:
                mic.kill()
            except Exception:
                pass

        print("Zeno stopped.")


if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print()
        print("🛑 Zeno stopped.")

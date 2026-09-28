import asyncio
import subprocess
from google import genai
from google.genai import types
from config.secrets import GEMINI_API_KEY

async def main():
    print("🎤 Recording 5 seconds...")

    audio = subprocess.run(
        ["arecord", "-D", "plughw:1,0", "-f", "S16_LE",
         "-r", "16000", "-c", "1", "-d", "5", "-t", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    ).stdout

    print("Recorded:", len(audio), "bytes")
    print("Connecting...")

    client = genai.Client(api_key=GEMINI_API_KEY)

    async with client.aio.live.connect(
        model="gemini-3.8-live",
        config=types.LiveConnectConfig(
            response_modalities=["AUDIO"]
        )
    ) as session:

        print("🟢 Connected")
        print("Sending audio...")

        await session.send_realtime_input(
            audio=types.Blob(
                data=audio,
                mime_type="audio/pcm;rate=16000"
            )
        )

        await session.send_realtime_input(
            audio_stream_end=True
        )

        print("Waiting for Zeno...")

        audio_data = b""

        async for response in session.receive():

            if response.server_content:
                content = response.server_content

                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data:
                            audio_data += part.inline_data.data

                if content.turn_complete:
                    break

        print("Received:", len(audio_data), "bytes")

        with open("voice/test_response.raw", "wb") as f:
            f.write(audio_data)

        print("✅ Gemini audio saved.")


asyncio.run(main())
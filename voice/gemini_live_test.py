import asyncio
import subprocess
import os
import sys
import shutil
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

# Ensure UTF-8 output encoding across Windows and Linux
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from google import genai
from google.genai import types
from google.genai import errors
from config.secrets import GEMINI_API_KEY


# ==========================================
# CONFIGURATION
# ==========================================

PRIMARY_MODEL = "gemini-3.8-live"
FALLBACK_MODEL = "gemini-3.1-flash-live-preview"

# ALSA device hardware addresses (frozen for Raspberry Pi)
MIC_ALSA = "plughw:1,0"
SPEAKER_ALSA = "plughw:1,1"

SAMPLE_RATE_IN = 16000     # 16 kHz mono S16_LE
SAMPLE_RATE_OUT = 24000    # 24 kHz mono S16_LE (Gemini Live native audio)
CHUNK_SIZE = 1280          # 40ms chunks @ 16kHz S16_LE

HAS_ARECORD = shutil.which("arecord") is not None
HAS_APLAY = shutil.which("aplay") is not None

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


# ==========================================
# AUDIO HAL (HARDWARE ABSTRACTION LAYER)
# ==========================================

class AudioHAL:
    """Manages audio capture and playback for both Raspberry Pi ALSA and local dev."""

    def __init__(self):
        self.use_alsa = HAS_ARECORD and HAS_APLAY
        if not self.use_alsa:
            print("ℹ️ ALSA not detected (running in fallback/sounddevice mode).")

    def start_microphone(self):
        if self.use_alsa:
            return subprocess.Popen(
                [
                    "arecord",
                    "-D", MIC_ALSA,
                    "-f", "S16_LE",
                    "-r", str(SAMPLE_RATE_IN),
                    "-c", "1",
                    "-t", "raw",
                    "-q",
                    "-"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
        else:
            # Fallback for Windows/local dev
            try:
                import sounddevice as sd
                import queue
                q = queue.Queue()

                def callback(indata, frames, time_info, status):
                    q.put(bytes(indata))

                stream = sd.RawInputStream(
                    samplerate=SAMPLE_RATE_IN,
                    blocksize=CHUNK_SIZE // 2,
                    channels=1,
                    dtype='int16',
                    callback=callback
                )
                stream.start()

                class SDMicWrapper:
                    def __init__(self, stream, q):
                        self.stream = stream
                        self.q = q
                    def read(self, size):
                        try:
                            return self.q.get(timeout=0.2)
                        except Exception:
                            return b"\x00" * size
                    def terminate(self):
                        try:
                            self.stream.stop()
                            self.stream.close()
                        except Exception:
                            pass
                    def kill(self):
                        self.terminate()
                    def wait(self, timeout=None):
                        pass

                return SDMicWrapper(stream, q)
            except Exception as e:
                print(f"⚠️ Simulated mic fallback active (error initializing sounddevice: {e})")
                class DummyMic:
                    def read(self, size):
                        time.sleep(0.04)
                        return b"\x00" * size
                    def terminate(self): pass
                    def kill(self): pass
                    def wait(self, timeout=None): pass
                return DummyMic()

    def play_audio(self, audio: bytes):
        if not audio:
            return

        if self.use_alsa:
            speaker = subprocess.Popen(
                [
                    "aplay",
                    "-D", SPEAKER_ALSA,
                    "-f", "S16_LE",
                    "-r", str(SAMPLE_RATE_OUT),
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
        else:
            try:
                import sounddevice as sd
                import numpy as np
                audio_np = np.frombuffer(audio, dtype=np.int16)
                sd.play(audio_np, samplerate=SAMPLE_RATE_OUT)
                sd.wait()
            except Exception as e:
                print(f"Dev playback simulation ({len(audio)} bytes received, {e})")


audio_hal = AudioHAL()


# ==========================================
# ASYNC WORKERS
# ==========================================

async def microphone_loop(session, mic, listening, stream_active):
    """
    Streams microphone chunks to Gemini Live.
    When listening is clear (during Zeno's playback), reads chunks
    to avoid ALSA buffer overrun but does not transmit them.
    """
    print("🎤 Microphone streaming started...")

    while True:
        try:
            if hasattr(mic, "stdout"):
                data = await asyncio.to_thread(mic.stdout.read, CHUNK_SIZE)
            else:
                data = await asyncio.to_thread(mic.read, CHUNK_SIZE)

            if not data:
                print("⚠️ Microphone stream closed.")
                return

            if listening.is_set():
                stream_active["user_streaming"] = True
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=data,
                        mime_type=f"audio/pcm;rate={SAMPLE_RATE_IN}"
                    )
                )
            else:
                # If Zeno is speaking or listening was paused, do not forward audio.
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ Microphone loop warning: {e}")
            break


async def receive_loop(session, listening, stream_active, session_state):
    """
    Receives transcription, native audio, and turn signaling from Gemini Live.
    Handles session resumption tokens and audio stream boundaries.
    """
    print("🔊 Gemini Live receiver active...")
    audio_buffer = bytearray()
    first_chunk_of_turn = True

    async for response in session.receive():
        # 1. Check for session resumption update
        if response.session_resumption_update:
            update = response.session_resumption_update
            if update.resumable and update.new_handle:
                session_state["resumption_handle"] = update.new_handle

        # 2. Check for server go-away
        if response.go_away:
            print("⚠️ Server sent GoAway signal. Session will reconnect.")
            break

        # 3. Check for voice activity signals
        if response.voice_activity:
            va_type = response.voice_activity.voice_activity_type
            if str(va_type).endswith("ACTIVITY_START"):
                print("🎙️ [VAD] User started speaking...", flush=True)
            elif str(va_type).endswith("ACTIVITY_END"):
                print("🎙️ [VAD] User paused/finished speaking.", flush=True)

        if not response.server_content:
            continue

        content = response.server_content

        # -----------------------------
        # USER TRANSCRIPTION
        # -----------------------------
        if content.input_transcription and content.input_transcription.text:
            user_text = content.input_transcription.text.strip()
            if user_text:
                print(f"\n👤 YOU: {user_text}", flush=True)

        # -----------------------------
        # GEMINI RESPONSE
        # -----------------------------
        if content.model_turn:
            # Model started outputting turn: finalize the previous input stream
            if first_chunk_of_turn and stream_active.get("user_streaming"):
                try:
                    await session.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass
                stream_active["user_streaming"] = False
                first_chunk_of_turn = False

            for part in content.model_turn.parts:
                if part.inline_data:
                    audio_buffer.extend(part.inline_data.data)

                if part.text:
                    print(f"🤖 ZENO: {part.text}", end="", flush=True)

        # -----------------------------
        # TURN COMPLETE
        # -----------------------------
        if content.turn_complete:
            print()
            first_chunk_of_turn = True

            if audio_buffer:
                audio = bytes(audio_buffer)
                print(f"🔊 Playing {len(audio)} bytes of native audio...")

                # Mute microphone to prevent acoustic echo during playback
                listening.clear()

                # Cleanly signal audio stream end before speaker output
                try:
                    await session.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass

                await asyncio.to_thread(audio_hal.play_audio, audio)

                # Playback complete: resume listening for next turn
                listening.set()
                session_state["turn_count"] += 1
                print(f"✅ Turn #{session_state['turn_count']} complete.")
                print(f"🎤 Listening for Turn #{session_state['turn_count'] + 1} (speak naturally)...")

            audio_buffer.clear()
            print()

    print("ℹ️ Server finished current response stream.")


# ==========================================
# MAIN SESSION CONTROLLER
# ==========================================

async def run_live_session():
    print()
    print("==================================================")
    print("         PROJECT T + ZENO LIVE (MULTI-TURN)       ")
    print("==================================================")
    print()

    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in environment or config/secrets.py")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    mic = audio_hal.start_microphone()

    listening = asyncio.Event()
    listening.set()

    stream_active = {"user_streaming": False}
    session_state = {
        "turn_count": 0,
        "resumption_handle": None
    }

    model_to_use = PRIMARY_MODEL
    max_reconnects = 5
    reconnect_delay = 2

    try:
        for attempt in range(1, max_reconnects + 1):
            print(f"\n🔄 Connecting to Gemini Live (Model: {model_to_use}, Attempt {attempt}/{max_reconnects})...")

            resumption_cfg = None
            if session_state.get("resumption_handle"):
                resumption_cfg = types.SessionResumptionConfig(
                    handle=session_state["resumption_handle"]
                )

            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=types.Content(
                    parts=[types.Part(text=SYSTEM)]
                ),
                input_audio_transcription=types.AudioTranscriptionConfig(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                session_resumption=resumption_cfg,
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(
                        disabled=False,
                        prefix_padding_ms=100,
                        silence_duration_ms=600
                    )
                )
            )

            try:
                async with client.aio.live.connect(
                    model=model_to_use,
                    config=config
                ) as session:
                    print("🟢 GEMINI LIVE CONNECTED")
                    if session_state["resumption_handle"]:
                        print("🔁 Session successfully resumed from previous state.")
                    print("--------------------------------------------------")
                    print("Zeno is continuously listening.")
                    print("Speak naturally. When Zeno answers, wait for playback.")
                    print("Then ask your next question without repeating 'Zeno'.")
                    print("Press Ctrl+C to stop.")
                    print("--------------------------------------------------\n")

                    mic_task = asyncio.create_task(
                        microphone_loop(session, mic, listening, stream_active)
                    )
                    receive_task = asyncio.create_task(
                        receive_loop(session, listening, stream_active, session_state)
                    )

                    # Monitor both tasks; if one terminates or fails, cancel the other
                    done, pending = await asyncio.wait(
                        [mic_task, receive_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass

                    # Check for exceptions in completed task
                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc

            except errors.APIError as e:
                err_msg = str(e)
                print(f"\n❌ Gemini API Error: {err_msg}")
                if (
                    "RESOURCE_EXHAUSTED" in err_msg
                    or "depleted" in err_msg.lower()
                    or "prepayment" in err_msg.lower()
                    or getattr(e, "code", None) == 402
                ):
                    print("\n⚠️ [CREDIT / BILLING NOTICE]")
                    print("Your Google AI Studio project prepayment credits are depleted.")
                    print("Please check: https://ai.studio/projects to review quota or add billing credits.")
                    print("Or configure a fresh API key in .env.")
                    break
                elif "NOT_FOUND" in err_msg and model_to_use == PRIMARY_MODEL:
                    print(f"⚠️ Primary model {PRIMARY_MODEL} not accessible, attempting fallback {FALLBACK_MODEL}...")
                    model_to_use = FALLBACK_MODEL
                    continue
                else:
                    print(f"Retrying connection in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)

            except Exception as e:
                err_str = str(e)
                print(f"\n⚠️ Session interrupted ({type(e).__name__}): {err_str}")
                if "1011" in err_str or "keepalive" in err_str or "closed" in err_str.lower():
                    print("🔁 Keepalive ping timeout or connection closed detected.")
                    if session_state["resumption_handle"]:
                        print("Attempting transparent session resumption...")
                    else:
                        print("Reconnecting Live session...")
                    await asyncio.sleep(reconnect_delay)
                else:
                    print(f"Reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)

    finally:
        print("\nStopping microphone...")
        try:
            mic.terminate()
            mic.wait(timeout=1)
        except Exception:
            try:
                mic.kill()
            except Exception:
                pass
        print("Zeno stopped cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(run_live_session())
    except KeyboardInterrupt:
        print("\n🛑 Zeno stopped by user.")

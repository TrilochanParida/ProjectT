import sys
import os
import subprocess
import asyncio
import json
import difflib
import time
import tempfile

# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ==========================================
# IMPORTS
# ==========================================

from faster_whisper import WhisperModel
from google import genai
from config.secrets import GEMINI_API_KEY
import edge_tts


# ==========================================
# FILE PATHS
# ==========================================

RAW_AUDIO = os.path.join(
    PROJECT_ROOT,
    "voice",
    "live_raw.wav"
)

MONO_AUDIO = os.path.join(
    PROJECT_ROOT,
    "voice",
    "live_mono.wav"
)

WAKE_RAW_AUDIO = os.path.join(
    PROJECT_ROOT,
    "voice",
    "wake_raw.wav"
)

WAKE_MONO_AUDIO = os.path.join(
    PROJECT_ROOT,
    "voice",
    "wake_mono.wav"
)

RESPONSE_MP3 = os.path.join(
    PROJECT_ROOT,
    "voice",
    "response.mp3"
)

RESPONSE_WAV = os.path.join(
    PROJECT_ROOT,
    "voice",
    "response.wav"
)

SETTINGS_FILE = os.path.join(
    PROJECT_ROOT,
    "config",
    "settings.json"
)


# ==========================================
# GEMINI
# ==========================================

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]


# ==========================================
# CONVERSATION SETTINGS
# ==========================================

# Maximum length of one question.
MAX_SPEECH_SECONDS = 20

# How much silence means the user has finished speaking.
END_SILENCE_SECONDS = 1.2

# How long Zeno waits for the next question
# before returning to standby.
CONVERSATION_TIMEOUT = 8

# Audio sampling.
SAMPLE_RATE = 48000


# ==========================================
# LOAD SETTINGS
# ==========================================

def load_settings():

    with open(
        SETTINGS_FILE,
        "r"
    ) as file:

        return json.load(file)


# ==========================================
# BASIC AUDIO RECORDING
# ==========================================

def record_audio(
    filename,
    duration
):

    subprocess.run([
        "arecord",
        "-D", "plughw:1,0",
        "-f", "S32_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "2",
        "-d", str(duration),
        filename
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)


# ==========================================
# CONVERT AUDIO
# ==========================================

def convert_audio(
    input_file,
    output_file
):

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-af", "pan=mono|c0=c0",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_file
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)


# ==========================================
# TRANSCRIBE
# ==========================================

def transcribe(
    model,
    audio_file
):

    segments, info = model.transcribe(
        audio_file,
        language="en",
        beam_size=1,
        vad_filter=True
    )

    text = ""

    for segment in segments:

        text += segment.text

    return text.strip()


# ==========================================
# WAKE WORD CHECK
# ==========================================

def is_wake_word(
    text,
    assistant_name
):

    text = text.lower().strip()

    assistant_name = (
        assistant_name
        .lower()
        .strip()
    )

    if not text:

        return False


    # Exact sentence

    if text == assistant_name:

        return True


    words = text.split()


    # Exact word

    for word in words:

        word = word.strip(
            ".,!?;:\"'()[]{}"
        )

        if word == assistant_name:

            return True


    # Fuzzy match for things like
    # "Jeno" / "Xeno"

    for word in words:

        word = word.strip(
            ".,!?;:\"'()[]{}"
        )

        similarity = difflib.SequenceMatcher(
            None,
            word,
            assistant_name
        ).ratio()

        if similarity >= 0.75:

            return True


    return False


# ==========================================
# WAIT FOR ZENO
# ==========================================

def wait_for_wake_word(
    model,
    assistant_name
):

    print("\n")
    print("======================================")
    print("          ZENO STANDBY MODE")
    print("======================================")

    print(
        f'Say "{assistant_name}" to start a conversation.'
    )

    print(
        "Press Ctrl+C to stop.\n"
    )


    while True:

        try:

            # ----------------------------------
            # Short listening window
            # ----------------------------------

            record_audio(
                WAKE_RAW_AUDIO,
                2
            )


            # ----------------------------------
            # Convert
            # ----------------------------------

            convert_audio(
                WAKE_RAW_AUDIO,
                WAKE_MONO_AUDIO
            )


            # ----------------------------------
            # Whisper
            # ----------------------------------

            text = transcribe(
                model,
                WAKE_MONO_AUDIO
            )


            # ----------------------------------
            # ONLY PRINT IF SOMETHING
            # INTERESTING WAS HEARD
            # ----------------------------------

            if text:

                if is_wake_word(
                    text,
                    assistant_name
                ):

                    print(
                        f'\n👂 Wake phrase detected: "{text}"'
                    )

                    print(
                        "\n🟢 ZENO AWAKENED!"
                    )

                    return True


        except KeyboardInterrupt:

            print(
                "\n\nStopping Project T..."
            )

            return False


        except Exception as error:

            print(
                f"\nWake detector error: {error}"
            )

            time.sleep(1)


# ==========================================
# START QUESTION RECORDING
#
# IMPORTANT:
#
# We use arecord's built-in silence
# detection here.
#
# It waits for speech to begin and
# stops after silence.
# ==========================================

def record_question():

    print("\n")
    print("======================================")
    print("          ZENO IS LISTENING")
    print("======================================")

    print(
        "🎤 Speak now..."
    )

    print(
        "Finish speaking normally."
    )

    print()


    subprocess.run([
        "arecord",

        "-D", "plughw:1,0",

        "-f", "S32_LE",

        "-r", str(SAMPLE_RATE),

        "-c", "2",

        # Start recording after sound
        "-B", "500000",

        # Stop after silence
        "-t", "wav",

        "-d", str(MAX_SPEECH_SECONDS),

        "-1",

        "-E", "silence",

        "-n",

        "-i", "0",

        "-f", "S32_LE",

        RAW_AUDIO

    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)


    # --------------------------------------
    # Convert
    # --------------------------------------

    convert_audio(
        RAW_AUDIO,
        MONO_AUDIO
    )


    return MONO_AUDIO


# ==========================================
# SIMPLE QUESTION RECORDING
#
# Fallback version.
#
# This is used if the above arecord
# silence options are not supported.
# ==========================================

def record_question_simple():

    print("\n")
    print("======================================")
    print("          ZENO IS LISTENING")
    print("======================================")

    print(
        "🎤 Speak your question..."
    )

    record_audio(
        RAW_AUDIO,
        MAX_SPEECH_SECONDS
    )

    convert_audio(
        RAW_AUDIO,
        MONO_AUDIO
    )

    return MONO_AUDIO


# ==========================================
# GEMINI
# ==========================================

def ask_gemini(
    client,
    assistant_name,
    user_text,
    conversation_history
):

    print(
        "\n🤖 Thinking..."
    )


    history_text = ""

    for item in conversation_history:

        history_text += (
            f"{item['role']}: "
            f"{item['text']}\n"
        )


    prompt = f"""
You are {assistant_name}, the intelligent AI brain
of Project T, a self-balancing intelligent companion robot.

You are having a natural ongoing conversation with the user.

Previous conversation:

{history_text}

The user's latest message is:

"{user_text}"

Answer naturally and conversationally.

Remember the context of the previous conversation.

Keep the answer concise because it will be spoken aloud
by a small robot.

Do not use markdown.
Do not use bullet points.
"""


    for model in GEMINI_MODELS:

        try:

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            return response.text.strip()


        except Exception as err:
            err_msg = str(err)
            if "RESOURCE_EXHAUSTED" in err_msg or "402" in err_msg:
                print(
                    f"⚠️ Quota/Prepayment exhausted on {model}: "
                    f"Please visit https://ai.studio/projects to check credits."
                )
            else:
                print(
                    f"{model} unavailable ({err}). "
                    f"Trying next model..."
                )


    return (
        "Sorry, I am unable to connect "
        "to my AI brain right now."
    )


# ==========================================
# TEXT TO SPEECH
# ==========================================

async def generate_speech(
    text,
    voice
):

    communicate = edge_tts.Communicate(
        text,
        voice
    )

    await communicate.save(
        RESPONSE_MP3
    )


def speak(
    text,
    voice
):

    print(
        "\n🔊 Zeno speaking..."
    )


    asyncio.run(
        generate_speech(
            text,
            voice
        )
    )


    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", RESPONSE_MP3,
        "-ar", "48000",
        "-ac", "2",
        "-sample_fmt", "s16",
        RESPONSE_WAV
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)


    subprocess.run([
        "aplay",
        "-D", "plughw:1,1",
        RESPONSE_WAV
    ],
    check=True)


# ==========================================
# CONVERSATION MODE
# ==========================================

def conversation_mode(
    whisper,
    client,
    assistant_name,
    voice
):

    print("\n")
    print("======================================")
    print("       ZENO CONVERSATION MODE")
    print("======================================")

    print(
        "You can now talk naturally."
    )

    print(
        f"You do NOT need to say {assistant_name} again."
    )

    print(
        f"Zeno will return to standby after "
        f"{CONVERSATION_TIMEOUT} seconds of inactivity."
    )

    print()


    # --------------------------------------
    # Conversation memory
    # --------------------------------------

    conversation_history = []


    # --------------------------------------
    # Greeting
    # --------------------------------------

    speak(
        f"Hello! I am {assistant_name}. "
        f"How can I help you?",
        voice
    )


    # ======================================
    # CONTINUOUS CONVERSATION
    # ======================================

    while True:

        try:

            print(
                "\n🎤 Waiting for your question..."
            )


            # --------------------------------
            # For now we use a short recording
            # window to detect the next speech.
            # --------------------------------

            record_audio(
                RAW_AUDIO,
                CONVERSATION_TIMEOUT
            )


            # --------------------------------
            # Convert
            # --------------------------------

            convert_audio(
                RAW_AUDIO,
                MONO_AUDIO
            )


            # --------------------------------
            # Whisper
            # --------------------------------

            print(
                "🧠 Understanding..."
            )


            user_text = transcribe(
                whisper,
                MONO_AUDIO
            )


            # --------------------------------
            # Nothing heard
            # --------------------------------

            if not user_text:

                print(
                    "\n🔵 No speech detected."
                )

                print(
                    "Returning to standby..."
                )

                return


            # --------------------------------
            # USER MESSAGE
            # --------------------------------

            print(
                "\n========== YOU SAID =========="
            )

            print(user_text)

            print(
                "=============================="
            )


            # --------------------------------
            # SAVE USER MESSAGE
            # --------------------------------

            conversation_history.append({
                "role": "User",
                "text": user_text
            })


            # --------------------------------
            # GEMINI
            # --------------------------------

            answer = ask_gemini(
                client,
                assistant_name,
                user_text,
                conversation_history
            )


            # --------------------------------
            # SAVE RESPONSE
            # --------------------------------

            conversation_history.append({
                "role": assistant_name,
                "text": answer
            })


            # --------------------------------
            # DISPLAY
            # --------------------------------

            print(
                "\n========== ZENO =========="
            )

            print(answer)

            print(
                "=========================="
            )


            # --------------------------------
            # SPEAK
            # --------------------------------

            speak(
                answer,
                voice
            )


            # --------------------------------
            # CONTINUE
            # --------------------------------

            print(
                "\n🔄 Conversation continues..."
            )


        except KeyboardInterrupt:

            print(
                "\n\nStopping conversation..."
            )

            return


        except Exception as error:

            print(
                f"\n❌ Conversation error: {error}"
            )

            return


# ==========================================
# MAIN
# ==========================================

def main():

    # --------------------------------------
    # SETTINGS
    # --------------------------------------

    settings = load_settings()


    assistant_name = settings[
        "assistant_name"
    ]

    voice = settings[
        "voice"
    ]

    wake_enabled = settings[
        "wake_word_enabled"
    ]


    # --------------------------------------
    # START
    # --------------------------------------

    print("\n")
    print("======================================")
    print(
        f"       PROJECT T + "
        f"{assistant_name.upper()}"
    )
    print("======================================")


    # --------------------------------------
    # LOAD WHISPER
    # --------------------------------------

    print("\nLoading Whisper...")

    whisper = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )

    print(
        "Whisper ready."
    )


    # --------------------------------------
    # GEMINI
    # --------------------------------------

    print(
        "Connecting to Gemini..."
    )

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


    print(
        f"\n{assistant_name.upper()} "
        f"IS READY 🤖"
    )


    # ======================================
    # MAIN ASSISTANT LOOP
    # ======================================

    while True:

        try:

            # --------------------------------
            # STANDBY
            # --------------------------------

            if wake_enabled:

                awakened = wait_for_wake_word(
                    whisper,
                    assistant_name
                )

                if not awakened:

                    break


            # --------------------------------
            # CONVERSATION
            # --------------------------------

            conversation_mode(
                whisper,
                client,
                assistant_name,
                voice
            )


            # --------------------------------
            # BACK TO STANDBY
            # --------------------------------

            print("\n")
            print(
                "🔵 Zeno is now in standby."
            )


        except KeyboardInterrupt:

            print(
                "\n\nStopping Project T..."
            )

            break


        except Exception as error:

            print(
                f"\n❌ Main error: {error}"
            )

            time.sleep(2)


# ==========================================
# ENTRY
# ==========================================

if __name__ == "__main__":

    main()
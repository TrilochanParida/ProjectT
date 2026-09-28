import sys
import os
import subprocess
import json
import difflib
import time

# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ==========================================
# IMPORT
# ==========================================

from faster_whisper import WhisperModel


# ==========================================
# FILES
# ==========================================

SETTINGS_FILE = os.path.join(
    PROJECT_ROOT,
    "config",
    "settings.json"
)

RAW_AUDIO = os.path.join(
    PROJECT_ROOT,
    "voice",
    "wake_raw.wav"
)

MONO_AUDIO = os.path.join(
    PROJECT_ROOT,
    "voice",
    "wake_mono.wav"
)


# ==========================================
# LOAD SETTINGS
# ==========================================

def load_settings():

    with open(SETTINGS_FILE, "r") as file:
        return json.load(file)


# ==========================================
# RECORD SHORT WAKE AUDIO
# ==========================================

def record_wake_audio():

    subprocess.run([
        "arecord",
        "-D", "plughw:1,0",
        "-f", "S32_LE",
        "-r", "48000",
        "-c", "2",
        "-d", "2",
        RAW_AUDIO
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)


# ==========================================
# CONVERT AUDIO
# ==========================================

def convert_audio():

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", RAW_AUDIO,
        "-af", "pan=mono|c0=c0",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        MONO_AUDIO
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)


# ==========================================
# TRANSCRIBE
# ==========================================

def transcribe(model):

    segments, info = model.transcribe(
        MONO_AUDIO,
        language="en",
        beam_size=1,
        vad_filter=True
    )

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()


# ==========================================
# CHECK NAME
# ==========================================

def is_wake_word(
    text,
    assistant_name
):

    text = text.lower().strip()
    assistant_name = assistant_name.lower().strip()

    if not text:
        return False

    # --------------------------------------
    # Exact match
    # --------------------------------------

    if text == assistant_name:

        return True


    # --------------------------------------
    # Name inside sentence
    # --------------------------------------

    words = text.split()

    for word in words:

        word = word.strip(
            ".,!?;:\"'()[]{}"
        )

        if word == assistant_name:

            return True


    # --------------------------------------
    # Fuzzy matching
    # --------------------------------------

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
        f'Say "{assistant_name}" to wake me.'
    )

    print(
        "Press Ctrl+C to stop.\n"
    )


    while True:

        try:

            print(
                "🟢 Standby... listening..."
            )

            # --------------------------------
            # Record
            # --------------------------------

            record_wake_audio()


            # --------------------------------
            # Convert
            # --------------------------------

            convert_audio()


            # --------------------------------
            # Whisper
            # --------------------------------

            text = transcribe(model)


            if text:

                print(
                    f'👂 Heard: "{text}"'
                )


            # --------------------------------
            # Check wake word
            # --------------------------------

            if is_wake_word(
                text,
                assistant_name
            ):

                print("\n")
                print(
                    "🟢🟢🟢 ZENO AWAKENED! 🟢🟢🟢"
                )

                print(
                    f"Wake word detected: "
                    f"{assistant_name}"
                )

                print()

                return True


        except KeyboardInterrupt:

            print(
                "\n\nStopping wake-word detector..."
            )

            return False


        except Exception as error:

            print(
                f"\nWake detector error: {error}"
            )

            time.sleep(1)


# ==========================================
# MAIN
# ==========================================

def main():

    settings = load_settings()

    assistant_name = settings[
        "assistant_name"
    ]


    # --------------------------------------
    # LOAD WHISPER
    # --------------------------------------

    print("\nLoading Whisper...")

    whisper = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )

    print("Whisper ready.")


    # --------------------------------------
    # START STANDBY
    # --------------------------------------

    wait_for_wake_word(
        whisper,
        assistant_name
    )


# ==========================================
# ENTRY
# ==========================================

if __name__ == "__main__":

    main()
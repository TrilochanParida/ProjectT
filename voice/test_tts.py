import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from voice.text_to_speech import speak


speak(
    "Hello! I am Project T. "
    "My voice system is working."
)

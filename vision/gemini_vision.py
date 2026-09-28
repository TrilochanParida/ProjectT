import os
import sys
import cv2
from PIL import Image
from io import BytesIO
from typing import Optional

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from google import genai
from google.genai import types
from config.secrets import GEMINI_API_KEY


class GeminiVision:
    """
    Cloud Multimodal Vision Provider for Scene Description, OCR, and Visual Q&A.
    Uses gemini-3.8-flash (with gemini-3.6-flash fallback).
    """

    MODELS = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash"
    ]

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError("GEMINI_API_KEY not configured.")
        self.client = genai.Client(api_key=key)

    def _frame_to_pil(self, frame_bgr) -> Image.Image:
        """Converts OpenCV BGR image to PIL RGB Image."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def query_image(self, frame_bgr, prompt: str) -> str:
        """
        Sends frame and prompt to Gemini Vision and returns the text response.
        """
        pil_image = self._frame_to_pil(frame_bgr)

        system_instruction = (
            "You are Zeno, the companion robot's vision perception system. "
            "Describe concisely what you observe in 1-2 natural sentences suitable "
            "for speaking aloud by a companion robot."
        )

        for model in self.MODELS:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=[
                        pil_image,
                        f"{system_instruction}\n\nUser Question: {prompt}"
                    ]
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                # Try next model in list
                continue

        return "I'm sorry, I was unable to analyze the image right now."

    def describe_scene(self, frame_bgr) -> str:
        """Answers: 'Zeno, what do you see in front of you?'"""
        return self.query_image(
            frame_bgr,
            "Describe the scene and any people or main objects directly in front of you."
        )

    def read_text_ocr(self, frame_bgr) -> str:
        """Answers: 'Zeno, read what is written on that sign / paper.'"""
        return self.query_image(
            frame_bgr,
            "Read and transcribe any visible text, signs, labels, or writing in the image."
        )

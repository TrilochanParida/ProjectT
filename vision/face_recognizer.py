import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class DetectedFace:
    """Represents a detected face with tracking metrics."""
    bbox: Tuple[int, int, int, int]       # (x, y, w, h)
    confidence: float                     # Detection confidence [0.0 - 1.0]
    landmarks: np.ndarray                 # 5 facial landmarks
    aligned_face: Optional[np.ndarray]    # 112x112 aligned face crop
    embedding: Optional[np.ndarray]       # 128-dimensional feature embedding
    center_offset_x: float                # Normalized horizontal offset [-1.0 (left) to +1.0 (right)]
    relative_size: float                  # Face box area as fraction of frame area


class FaceEngine:
    """
    Lightweight Face Detection and Recognition Engine for Raspberry Pi 3.
    Backed by OpenCV YuNet (Detector) and SFace (Feature Recognizer).
    """

    COSINE_THRESHOLD = 0.363  # OpenCV SFace recommended cosine similarity threshold

    def __init__(
        self,
        model_dir: str = "vision/models",
        input_size: Tuple[int, int] = (640, 480),
        score_threshold: float = 0.65,
        nms_threshold: float = 0.3
    ):
        self.model_dir = model_dir
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self.detector = None
        self.recognizer = None
        self.haar_cascade = None

        self._load_models()

    def _load_models(self):
        yunet_path = os.path.join(self.model_dir, "face_detection_yunet_2023mar.onnx")
        sface_path = os.path.join(self.model_dir, "face_recognition_sface_2021dec.onnx")

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Auto-download YuNet if missing
        if not os.path.exists(yunet_path):
            print("📥 FaceEngine: Downloading YuNet face detection model (232 KB)...")
            try:
                import urllib.request
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                urllib.request.urlretrieve(url, yunet_path)
            except Exception as e:
                print(f"⚠️ FaceEngine: Failed to download YuNet: {e}")

        # Auto-download SFace if missing
        if not os.path.exists(sface_path):
            print("📥 FaceEngine: Downloading SFace face recognition model (38 MB)...")
            try:
                import urllib.request
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
                urllib.request.urlretrieve(url, sface_path)
            except Exception as e:
                print(f"⚠️ FaceEngine: Failed to download SFace: {e}")

        # 1. Initialize YuNet
        if os.path.exists(yunet_path):
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=yunet_path,
                    config="",
                    input_size=self.input_size,
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=5000
                )
                print("🧠 FaceEngine: YuNet face detector loaded.")
            except Exception as e:
                print(f"⚠️ FaceEngine: Failed to load YuNet ({e})")

        # 2. Initialize SFace
        if os.path.exists(sface_path):
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=sface_path,
                    config=""
                )
                print("🧠 FaceEngine: SFace face recognizer loaded.")
            except Exception as e:
                print(f"⚠️ FaceEngine: Failed to load SFace ({e})")

        # 3. Haar Cascade fallback
        if self.detector is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.haar_cascade = cv2.CascadeClassifier(cascade_path)
                print("ℹ️ FaceEngine: Loaded Haar Cascade fallback.")

    def set_input_size(self, width: int, height: int):
        """Update detection input size dynamically."""
        self.input_size = (width, height)
        if self.detector is not None:
            self.detector.setInputSize((width, height))

    def detect_faces(self, frame: np.ndarray, extract_features: bool = True) -> List[DetectedFace]:
        """
        Detects all faces in the given BGR frame.
        Computes bounding box, landmarks, normalized offsets, and embeddings.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if (w, h) != self.input_size:
            self.set_input_size(w, h)

        results: List[DetectedFace] = []

        # --- Detection via YuNet ---
        if self.detector is not None:
            _, faces = self.detector.detect(frame)
            if faces is not None:
                for face in faces:
                    box = [int(v) for v in face[0:4]]
                    x, y, fw, fh = box
                    conf = float(face[-1])
                    landmarks = face[4:14].reshape((5, 2))

                    # Calculate target tracking metrics for robot Follow-Me
                    face_center_x = x + (fw / 2.0)
                    center_offset_x = (face_center_x - (w / 2.0)) / (w / 2.0)
                    relative_size = (fw * fh) / float(w * h)

                    aligned_face = None
                    embedding = None

                    if extract_features and self.recognizer is not None:
                        try:
                            aligned_face = self.recognizer.alignCrop(frame, face)
                            feat = self.recognizer.feature(aligned_face)
                            embedding = feat.flatten()
                        except Exception:
                            pass

                    results.append(DetectedFace(
                        bbox=(x, y, fw, fh),
                        confidence=conf,
                        landmarks=landmarks,
                        aligned_face=aligned_face,
                        embedding=embedding,
                        center_offset_x=float(np.clip(center_offset_x, -1.0, 1.0)),
                        relative_size=float(relative_size)
                    ))
            return results

        # --- Fallback: Haar Cascade ---
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                face_center_x = x + (fw / 2.0)
                center_offset_x = (face_center_x - (w / 2.0)) / (w / 2.0)
                relative_size = (fw * fh) / float(w * h)

                results.append(DetectedFace(
                    bbox=(x, y, fw, fh),
                    confidence=0.85,
                    landmarks=np.array([]),
                    aligned_face=None,
                    embedding=None,
                    center_offset_x=float(np.clip(center_offset_x, -1.0, 1.0)),
                    relative_size=float(relative_size)
                ))

        return results

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two 128D embeddings.
        Returns score between -1.0 and 1.0 (Scores > 0.363 typically indicate same identity).
        """
        if emb1 is None or emb2 is None:
            return 0.0

        if self.recognizer is not None:
            # OpenCV's internal cosine matching
            f1 = emb1.reshape(1, -1).astype(np.float32)
            f2 = emb2.reshape(1, -1).astype(np.float32)
            return float(self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE))

        # Direct numpy cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

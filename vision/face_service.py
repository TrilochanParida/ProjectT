import os
import sys
import time
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vision.camera_hal import CameraHAL
from vision.face_recognizer import FaceEngine, DetectedFace
from vision.profile_manager import ProfileManager


class FaceRecognitionService:
    """
    High-Level Face Recognition and Visual Person Tracking Service for Zeno.
    Coordinates Camera HAL, Local Neural Embedding Engine, and Profile Memory Store.
    """

    def __init__(self, camera: Optional[CameraHAL] = None):
        self.camera = camera or CameraHAL(width=640, height=480)
        self.engine = FaceEngine()
        self.profiles = ProfileManager()
        self.last_greeting_times: Dict[str, float] = {}
        self.greeting_cooldown = 30.0  # Greet known person at most once every 30 seconds

    def process_live_frame(self, draw_overlay: bool = True) -> Tuple[np.ndarray, List[dict]]:
        """
        Captures one frame, detects all faces, matches identities,
        and annotates tracking data.
        Returns: (annotated_frame, list_of_detected_people_info)
        """
        frame = self.camera.capture_frame()
        if frame is None or frame.size == 0:
            return frame, []

        detected_faces = self.engine.detect_faces(frame, extract_features=True)
        people_info = []

        for face in detected_faces:
            x, y, w, h = face.bbox
            name, similarity, greeting = (None, 0.0, None)

            if face.embedding is not None:
                name, similarity, greeting = self.profiles.identify_person(face.embedding)

            is_known = name is not None
            display_label = f"{name} ({similarity:.2f})" if is_known else f"Unknown ({face.confidence:.2f})"
            color = (0, 255, 0) if is_known else (0, 165, 255)  # Green for known, Orange for unknown

            # Tracking telemetry for robot turning and follow-me
            info = {
                "name": name,
                "is_known": is_known,
                "similarity": similarity,
                "confidence": face.confidence,
                "bbox": face.bbox,
                "center_offset_x": face.center_offset_x,  # [-1.0 (left), +1.0 (right)]
                "relative_size": face.relative_size,      # Distance estimator (larger = closer)
                "custom_greeting": greeting
            }
            people_info.append(info)

            if draw_overlay:
                # Bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Label banner
                cv2.rectangle(frame, (x, max(0, y - 24)), (x + w, y), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    display_label,
                    (x + 4, max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

                # Draw tracking crosshair
                cx, cy = int(x + w / 2), int(y + h / 2)
                cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

        return frame, people_info

    def check_greeting(self, person_name: str, custom_greeting: Optional[str] = None) -> Optional[str]:
        """
        Checks if person should be greeted (respecting cooldown).
        Returns greeting message if due, else None.
        """
        now = time.time()
        last_greet = self.last_greeting_times.get(person_name, 0)

        if (now - last_greet) > self.greeting_cooldown:
            self.last_greeting_times[person_name] = now
            return custom_greeting or f"Hello {person_name}, wonderful to see you!"
        return None

    def register_person_from_camera(self, name: str, samples: int = 3) -> bool:
        """
        Captures multiple face samples of the person in front of the camera,
        averages / stores their embeddings into profiles.json with consent.
        """
        print(f"📸 Registering '{name}'... Please look directly at the camera.")
        captured_embeddings = []

        for i in range(samples):
            time.sleep(0.4)
            frame = self.camera.capture_frame()
            faces = self.engine.detect_faces(frame, extract_features=True)

            if not faces:
                print(f"⚠️ No face detected in sample {i+1}/{samples}. Retrying...")
                continue

            # Pick the largest/most prominent face
            largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            if largest_face.embedding is not None:
                captured_embeddings.append(largest_face.embedding)
                print(f"✅ Captured sample {len(captured_embeddings)}/{samples}")

        if not captured_embeddings:
            print("❌ Registration failed: No clear face samples captured.")
            return False

        # Average embeddings across captures
        avg_embedding = np.mean(captured_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding /= norm

        success = self.profiles.register_person(name, avg_embedding)
        return success

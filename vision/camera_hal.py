import os
import sys
import time
import shutil
import subprocess
import cv2
import numpy as np

# Ensure UTF-8 output encoding across Windows and Linux
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class CameraHAL:
    """
    Hardware Abstraction Layer for Raspberry Pi Camera Module 3 Wide and Dev Webcams.
    Prioritizes:
    1. Picamera2 (Official Pi Camera Module 3 driver on Debian 13/Bookworm)
    2. rpicam-still / libcamera-still CLI pipeline
    3. OpenCV VideoCapture(0) (USB camera or dev webcam)
    4. Dummy test frame generator (for headless CI / headless testing)
    """

    def __init__(self, width=640, height=480, framerate=15):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.backend = None
        self.cam = None
        self._init_camera()

    def _init_camera(self):
        # 1. Try Picamera2 (Raspberry Pi native on Linux)
        if sys.platform != "win32":
            try:
                import importlib
                picam2_module = importlib.import_module("picamera2")
                Picamera2 = getattr(picam2_module, "Picamera2")
                self.cam = Picamera2()
                config = self.cam.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "BGR888"}
                )
                self.cam.configure(config)
                self.cam.start()
                self.backend = "picamera2"
                print("📷 Camera HAL: Initialized with Picamera2 (Pi Camera Module 3)")
                return
            except Exception:
                pass

        # 2. Try OpenCV VideoCapture
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.framerate)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cam = cap
                    self.backend = "opencv"
                    print("📷 Camera HAL: Initialized with OpenCV VideoCapture(0)")
                    return
                else:
                    cap.release()
        except Exception:
            pass

        # 3. Check for rpicam-still
        if shutil.which("rpicam-still") or shutil.which("libcamera-still"):
            self.backend = "rpicam-cli"
            print("📷 Camera HAL: Initialized with rpicam-still CLI snapshot mode")
            return

        # 4. Fallback: Simulated test pattern
        self.backend = "mock"
        print("ℹ️ Camera HAL: Running in simulated test mode (no physical camera detected).")

    def capture_frame(self) -> np.ndarray:
        """Captures a single BGR frame."""
        if self.backend == "picamera2":
            return self.cam.capture_array()

        elif self.backend == "opencv":
            ret, frame = self.cam.read()
            if ret and frame is not None:
                return frame
            # Return blank frame if read failed
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        elif self.backend == "rpicam-cli":
            cli = "rpicam-still" if shutil.which("rpicam-still") else "libcamera-still"
            os.makedirs("vision/snapshots", exist_ok=True)
            temp_path = "vision/snapshots/temp_snapshot.jpg"
            try:
                subprocess.run(
                    [
                        cli,
                        "-t", "200",
                        "--width", str(self.width),
                        "--height", str(self.height),
                        "-o", temp_path,
                        "-n"
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                frame = cv2.imread(temp_path)
                if frame is not None:
                    return frame
            except Exception:
                pass
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        else:
            # Generate simulated frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Zeno Simulated Camera",
                (30, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            return frame

    def save_snapshot(self, output_path: str) -> bool:
        """Captures a frame and saves it directly to disk."""
        frame = self.capture_frame()
        if frame is not None and frame.size > 0:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            return cv2.imwrite(output_path, frame)
        return False

    def release(self):
        """Releases camera resources cleanly."""
        if self.backend == "picamera2" and self.cam is not None:
            try:
                self.cam.stop()
                self.cam.close()
            except Exception:
                pass
        elif self.backend == "opencv" and self.cam is not None:
            try:
                self.cam.release()
            except Exception:
                pass
        self.cam = None

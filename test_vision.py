import os
import sys
import time
import argparse
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from vision.face_service import FaceRecognitionService
from vision.gemini_vision import GeminiVision


def main():
    parser = argparse.ArgumentParser(description="Project T + Zeno Vision & Face Recognition")
    parser.add_argument("--register", type=str, help="Register a new person by name")
    parser.add_argument("--list", action="store_true", help="List all registered people")
    parser.add_argument("--delete", type=str, help="Delete a registered person by name")
    parser.add_argument("--describe", action="store_true", help="Take snapshot and describe scene with Gemini")
    parser.add_argument("--ocr", action="store_true", help="Take snapshot and read text with Gemini OCR")
    parser.add_argument("--continuous", action="store_true", help="Run continuous recognition loop")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to process in test (default: 10)")
    args = parser.parse_args()

    print()
    print("==================================================")
    print("        PROJECT T — ZENO VISION & FACE SYSTEM     ")
    print("==================================================")
    print()

    service = FaceRecognitionService()

    # --- 1. LIST PROFILES ---
    if args.list:
        people = service.profiles.list_people()
        print(f"📋 Registered Profiles ({len(people)}):")
        for p in people:
            print(f"  • {p}")
        return

    # --- 2. DELETE PROFILE ---
    if args.delete:
        service.profiles.delete_person(args.delete)
        return

    # --- 3. REGISTER PERSON ---
    if args.register:
        name = args.register.strip()
        print(f"👤 Starting registration for: {name}")
        print("Stand in front of the camera...")
        success = service.register_person_from_camera(name, samples=3)
        if success:
            print(f"🎉 '{name}' is now registered in Zeno's local memory!")
        else:
            print(f"❌ Failed to register '{name}'.")
        return

    # --- 4. DESCRIBE SCENE (GEMINI VISION) ---
    if args.describe:
        print("📸 Capturing frame for scene analysis...")
        frame = service.camera.capture_frame()
        os.makedirs("vision/snapshots", exist_ok=True)
        cv2.imwrite("vision/snapshots/scene_query.jpg", frame)
        print("🧠 Sending to Gemini Vision...")
        try:
            gv = GeminiVision()
            desc = gv.describe_scene(frame)
            print("\n🤖 ZENO SEES:")
            print(desc)
        except Exception as e:
            print(f"❌ Vision query failed: {e}")
        return

    # --- 5. OCR TEXT READING ---
    if args.ocr:
        print("📸 Capturing frame for text reading (OCR)...")
        frame = service.camera.capture_frame()
        os.makedirs("vision/snapshots", exist_ok=True)
        cv2.imwrite("vision/snapshots/ocr_query.jpg", frame)
        print("🧠 Reading text with Gemini Vision...")
        try:
            gv = GeminiVision()
            text = gv.read_text_ocr(frame)
            print("\n📖 ZENO READS:")
            print(text)
        except Exception as e:
            print(f"❌ OCR query failed: {e}")
        return

    # --- 6. CONTINUOUS OR SAMPLE RECOGNITION LOOP ---
    print("🚀 Starting Face Recognition & Tracking...")
    print(f"Processing {'continuously (Press Ctrl+C to stop)' if args.continuous else f'{args.frames} test frames'}...\n")

    has_display = bool(os.environ.get("DISPLAY") or sys.platform == "win32")
    frame_idx = 0

    try:
        while True:
            frame_idx += 1
            annotated_frame, people = service.process_live_frame(draw_overlay=True)

            if people:
                for p in people:
                    name_str = p['name'] if p['is_known'] else 'Unknown'
                    print(
                        f"[Frame {frame_idx:03d}] 👤 Person: {name_str:<12} | "
                        f"Offset X: {p['center_offset_x']:+.2f} | "
                        f"Rel Size: {p['relative_size']:.3f} | "
                        f"Sim: {p['similarity']:.2f}"
                    )

                    # Trigger greeting if due
                    if p['is_known']:
                        greeting = service.check_greeting(p['name'], p['custom_greeting'])
                        if greeting:
                            print(f"\n🗣️ GREETING TRIGGERED: \"{greeting}\"\n")
            else:
                if frame_idx % 5 == 0:
                    print(f"[Frame {frame_idx:03d}] 👁️ Scanning... No face in view.")

            # Save snapshot every 10 frames
            if frame_idx % 10 == 0:
                os.makedirs("vision/snapshots", exist_ok=True)
                cv2.imwrite("vision/snapshots/latest_detection.jpg", annotated_frame)

            if not args.continuous and frame_idx >= args.frames:
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped face recognition loop.")
    finally:
        service.camera.release()
        cv2.destroyAllWindows()
        print("Camera released. Done.")


if __name__ == "__main__":
    main()

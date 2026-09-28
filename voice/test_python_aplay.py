import subprocess
import time
import os

speaker = subprocess.Popen(
    [
        "aplay",
        "-D", "plughw:1,1",
        "-f", "S16_LE",
        "-r", "24000",
        "-c", "1",
        "-t", "raw",
        "-"
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    bufsize=0
)

print("Started aplay. PID:", speaker.pid)

# 1 second of silence
audio = b"\x00" * (24000 * 2)

print("Writing silence...")
speaker.stdin.write(audio)
speaker.stdin.flush()

print("Write successful.")
print("Process status:", speaker.poll())

time.sleep(3)

print("After 3 seconds:")
print("Process status:", speaker.poll())

if speaker.poll() is None:
    print("✅ Python → aplay pipe is working.")
else:
    print("❌ aplay exited.")

speaker.stdin.close()

try:
    speaker.wait(timeout=2)
except subprocess.TimeoutExpired:
    speaker.terminate()

print("Done.")

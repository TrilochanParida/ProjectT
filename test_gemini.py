from google import genai
from config.secrets import GEMINI_API_KEY
import time

def main():
    print("Connecting to Gemini...")

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        "You are Project T, an intelligent AI companion robot. "
        "Introduce yourself to your creator in exactly two short sentences."
    )

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
    ]

    for model in models:
        print(f"\nTrying model: {model}")

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            print("\n========== PROJECT T ==========")
            print(response.text)
            print("================================")
            print(f"Successful model: {model}")
            return

        except Exception as e:
            print(f"Model failed: {e}")
            print("Trying next model...")
            time.sleep(2)

    print("\nAll Gemini models failed.")
    print("This is most likely a temporary API/service availability issue.")

if __name__ == "__main__":
    main()
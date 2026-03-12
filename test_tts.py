import sys

def main():
    try:
        import pyttsx3
    except Exception as e:
        print(f"pyttsx3 import failed: {e}")
        return 1

    try:
        engine = pyttsx3.init()
        driver = engine._driver.__class__.__name__ if hasattr(engine, "_driver") else "unknown"
        print(f"pyttsx3 init ok, driver={driver}")
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 0.9)
        engine.say("System started tracking")
        engine.runAndWait()
        print("TTS finished")
        return 0
    except Exception as e:
        print(f"TTS failed: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())

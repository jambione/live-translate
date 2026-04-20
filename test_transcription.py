"""
test_transcription.py — Standalone tests for Whisper transcription and LLM signal classification.
No workflows, no window automation — just audio → text → signal.

Usage
-----
  python test_transcription.py
"""

import time
import numpy as np
import ollama
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

# ========================= CONFIG =========================
LLM_MODEL      = "gemma2:2b"
SAMPLE_RATE    = 44100
TARGET_SR      = 16000
SILENCE_THRESHOLD = 0.009

# ========================= INIT =========================
print("Loading Whisper 'small' model...")
whisper = WhisperModel("small", device="cpu", compute_type="int8")
print("✅ Whisper ready.\n")

# ========================= HELPERS =========================

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: list[tuple[str, str, str]] = []   # (label, expected, status)


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def run_llm(text: str) -> str:
    """Send text to Ollama and return the raw one-line response."""
    prompt = f"""You are a strict stock trading signal extractor.
Only respond if a clear stock ticker (3-5 uppercase letters like AAPL, TSLA, UNH, NVDA) is mentioned.

Reply with EXACTLY one line and nothing else:
- "buy TICKER" if buying is mentioned
- "sell TICKER" if selling is mentioned
- "watch TICKER" if watching, adding, or tracking is mentioned
- "NO ACTION" otherwise

Text: {text}"""

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_ctx": 2048, "num_predict": 40},
    )
    return resp["message"]["content"].strip()


def parse_signal(raw: str) -> tuple[str, str]:
    """Parse raw LLM output into (verb, ticker) or ('none', '')."""
    parts = raw.lower().split()
    if len(parts) >= 2 and parts[0] in ("buy", "sell", "watch"):
        verb   = parts[0]
        ticker = parts[1].upper()
        if 2 <= len(ticker) <= 5 and ticker.isalpha():
            return verb, ticker
    return "none", ""


# ========================= LLM SIGNAL TESTS =========================

LLM_TEST_CASES = [
    # (phrase,                                            expected_verb, expected_ticker)
    ("I want to watch Apple, ticker AAPL",                "watch", "AAPL"),
    ("Let's buy some Tesla, TSLA looks strong",           "buy",   "TSLA"),
    ("Time to sell UNH, get out now",                     "sell",  "UNH"),
    ("NVDA is worth keeping an eye on",                   "watch", "NVDA"),
    ("Add Microsoft MSFT to the list",                    "watch", "MSFT"),
    ("Go ahead and buy AMZN",                             "buy",   "AMZN"),
    ("Sell everything, dump GOOGL",                       "sell",  "GOOGL"),
    ("The weather looks great today",                     "none",  ""),
    ("I had a great lunch",                               "none",  ""),
    ("UNH is interesting, I want to track it",            "watch", "UNH"),
]


def test_llm_signals():
    section("LLM Signal Classification Tests")
    print(f"  Model: {LLM_MODEL}")
    print(f"  Running {len(LLM_TEST_CASES)} test cases...\n")

    for phrase, exp_verb, exp_ticker in LLM_TEST_CASES:
        t0  = time.time()
        raw = run_llm(phrase)
        ms  = int((time.time() - t0) * 1000)

        verb, ticker = parse_signal(raw)
        ok = (verb == exp_verb and ticker == exp_ticker)
        status = PASS if ok else FAIL

        label = f"{exp_verb.upper()} {exp_ticker}" if exp_verb != "none" else "NO ACTION"
        results.append((phrase[:45], label, status))

        mark = "✅" if ok else "❌"
        print(f"  {mark}  [{ms:4d}ms]  raw='{raw}'")
        if not ok:
            print(f"       expected: {exp_verb} {exp_ticker}  |  got: {verb} {ticker}")
        print(f"       \"{phrase}\"")
        print()


# ========================= LIVE TRANSCRIPTION TEST =========================

def pick_audio_device() -> int:
    p = pyaudio.PyAudio()
    print("\nAvailable audio input devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev["maxInputChannels"] > 0:
            tag = " ← LOOPBACK" if "loopback" in dev["name"].lower() else ""
            print(f"  {i:2d}: {dev['name']}{tag}")

    try:
        choice = input("\nEnter device index (Enter = default): ").strip()
        idx = int(choice) if choice else p.get_default_input_device_info()["index"]
    except Exception:
        idx = p.get_default_input_device_info()["index"]

    p.terminate()
    return idx


def capture_audio(device_index: int, duration: float) -> np.ndarray:
    """Record `duration` seconds of audio and return as a float32 array at TARGET_SR."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=2,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=int(SAMPLE_RATE * 0.5),
    )

    frames = []
    total_samples = int(SAMPLE_RATE * duration)
    collected = 0

    while collected < total_samples:
        chunk_size = min(int(SAMPLE_RATE * 0.5), total_samples - collected)
        data = stream.read(chunk_size, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.float32)
        frames.append(audio)
        collected += chunk_size

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.concatenate(frames)
    audio = audio.reshape(-1, 2).mean(axis=1)          # stereo → mono
    audio = resample_poly(audio, TARGET_SR, SAMPLE_RATE, window=("kaiser", 5.0))
    return audio.astype(np.float32)


def transcribe(audio: np.ndarray) -> str:
    segments, _ = whisper.transcribe(
        audio,
        language="en",
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def test_live_transcription(device_index: int, duration: float = 6.0):
    section(f"Live Transcription Test  ({duration}s)")
    print(f"  🎙️  Recording for {duration}s — speak a ticker phrase now...")
    print(f"      e.g. 'watch Apple AAPL' / 'buy Tesla' / 'sell UNH'\n")

    audio = capture_audio(device_index, duration)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"  Audio RMS level: {rms:.4f}  (silence threshold: {SILENCE_THRESHOLD})")

    if rms < SILENCE_THRESHOLD:
        print("  ⚠️  Audio level very low — check your input device.")

    t0   = time.time()
    text = transcribe(audio)
    ms   = int((time.time() - t0) * 1000)

    print(f"\n  Whisper ({ms}ms): \"{text}\"")

    if text:
        t0  = time.time()
        raw = run_llm(text)
        ms  = int((time.time() - t0) * 1000)
        verb, ticker = parse_signal(raw)

        print(f"  LLM    ({ms}ms): {raw}")
        if verb != "none":
            print(f"\n  🤖 Signal → {verb.upper()} {ticker}")
        else:
            print(f"\n  ℹ️  No trading signal detected.")
    else:
        print("  ⚠️  No transcription produced.")


# ========================= SUMMARY =========================

def print_summary():
    section("Results Summary")
    if not results:
        print("  No tests recorded.")
        return

    max_label = max(len(r[0]) for r in results)
    passed  = sum(1 for _, _, s in results if s == PASS)
    failed  = sum(1 for _, _, s in results if s == FAIL)

    for phrase, expected, status in results:
        print(f"  {status}  {phrase:<{max_label}}  → {expected}")

    print(f"\n  {passed}/{len(results)} passed  |  {failed} failed")


# ========================= MENU =========================

MENU = """
  1  LLM signal tests     — run all phrase → signal test cases
  2  Live transcription   — record mic/loopback and transcribe
  3  Run both
  q  Quit
"""


def main():
    print(f"\n🧪  Transcription + LLM Signal Test Suite")
    print(f"    Model: {LLM_MODEL}")
    print(MENU)

    device_index = None

    while True:
        choice = input("Choice (1/2/3/q): ").strip().lower()

        if choice in ("q", "quit"):
            break
        elif choice == "1":
            test_llm_signals()
            print_summary()
        elif choice in ("2", "3"):
            if device_index is None:
                device_index = pick_audio_device()

            if choice == "3":
                test_llm_signals()

            dur_input = input("\n  Recording duration in seconds [Enter = 6]: ").strip()
            try:
                dur = float(dur_input) if dur_input else 6.0
            except ValueError:
                dur = 6.0

            again = True
            while again:
                test_live_transcription(device_index, dur)
                print_summary()
                again = input("\n  Record again? [y/n]: ").strip().lower() in ("y", "yes")
        else:
            print("  Invalid choice.")

        print()


if __name__ == "__main__":
    main()

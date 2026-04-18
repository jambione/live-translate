import pyaudiowpatch as pyaudio
import numpy as np
import time
import ollama
import threading
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
from queue import Queue

# ========================= CONFIG =========================
DEVICE_INDEX = 10
LLM_MODEL = "phi3:mini"          # or "tinyllama" for even lighter
LLM_INTERVAL = 4.0               # LLM runs every 4 seconds

print("Loading Whisper 'base' model...")
whisper = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Whisper ready.\n")

print(f"Loading LLM: {LLM_MODEL}...")
print("✅ LLM ready.\n")

p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paFloat32,
    channels=2,
    rate=44100,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=44100
)
print(f"✅ Stream opened on device {DEVICE_INDEX}. Play your video / audio now.\n")

buffer = []
last_llm_time = 0.0
llm_queue = Queue()
running = True

def llm_worker():
    global running
    while running:
        try:
            text = llm_queue.get(timeout=1.0)
            if text is None:
                break
            if time.time() - last_llm_time < LLM_INTERVAL:
                continue

            # Specialized prompt for stock ticker buy/sell detection
            prompt = f"""You are a stock trading signal extractor.
Scan the text and identify any stock ticker (3-5 uppercase letters).
Determine if the speaker is talking about BUYING or SELLING the position.
Only reply with one line in this exact format:
buy TICKER
sell TICKER
or exactly "NO ACTION"

Text: {text}"""

            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": 1024}
            )
            action = resp['message']['content'].strip()

            if action and action != "NO ACTION":
                print(f"🤖 TRADE SIGNAL: {action}")

            last_llm_time = time.time()
        except:
            pass

# Start LLM worker
llm_thread = threading.Thread(target=llm_worker, daemon=True)
llm_thread.start()

try:
    while running:
        data = stream.read(44100, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.float32)
        audio = audio.reshape(-1, 2).mean(axis=1)
        audio = resample_poly(audio, 16000, 44100, window=('kaiser', 5.0))

        if np.sqrt(np.mean(audio**2)) < 0.008:
            continue

        buffer.extend(audio)

        if len(buffer) >= 16000 * 2:
            chunk = np.array(buffer[:16000*2], dtype=np.float32)
            buffer = buffer[int(16000*1.0):]

            segments, _ = whisper.transcribe(
                chunk, language="en", vad_filter=False, beam_size=7, temperature=0.0
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

            if text and len(text) > 10:
                print(f"[{time.strftime('%H:%M:%S')}] {text}")

                # Send to LLM for stock signal extraction
                if time.time() - last_llm_time > LLM_INTERVAL:
                    llm_queue.put(text)
                    last_llm_time = time.time()

except KeyboardInterrupt:
    print("\nShutting down...")
    running = False
    llm_queue.put(None)
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("All processes terminated cleanly.")
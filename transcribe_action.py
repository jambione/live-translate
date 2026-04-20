import pyaudiowpatch as pyaudio
import numpy as np
import time
import re
import subprocess
import urllib.request
import ollama
import threading
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
from queue import Queue, Full
import pyperclip
from collections import deque

from workflows import workflow_add_wb, workflow_buy, workflow_sell_all, wb_watchlist_show, wb_watchlist_clear


# ========================= OLLAMA STARTUP CHECK =========================

def ensure_ollama_running(timeout: int = 20):
    """
    Ping the Ollama API. If it's not responding, launch `ollama serve`
    and wait up to `timeout` seconds for it to become ready.
    """
    url = "http://localhost:11434/api/tags"

    def is_ready() -> bool:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            return False

    if is_ready():
        print("✅ Ollama already running.\n")
        return

    print("⚡ Ollama not detected — starting ollama serve...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,  # no console popup on Windows
        )
    except Exception as e:
        print(f"❌ Could not start ollama serve: {e}")
        print("   Make sure Ollama is installed: https://ollama.com/download")
        raise SystemExit(1)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_ready():
            print("✅ Ollama ready.\n")
            return
        time.sleep(0.5)

    print(f"❌ Ollama did not respond within {timeout}s — aborting.")
    raise SystemExit(1)


ensure_ollama_running()


# ========================= TRANSCRIPT NORMALIZATION =========================

# NATO phonetic alphabet → letter mapping
_NATO = {
    "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D",
    "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H",
    "india": "I", "juliet": "J", "kilo": "K", "lima": "L",
    "mike": "M", "november": "N", "oscar": "O", "papa": "P",
    "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T",
    "uniform": "U", "victor": "V", "whiskey": "W", "xray": "X",
    "x-ray": "X", "yankee": "Y", "zulu": "Z",
}

# Pre-compiled pattern: 2–5 consecutive NATO words (case-insensitive)
_nato_word = "(?:" + "|".join(re.escape(w) for w in _NATO) + ")"
_NATO_PATTERN = re.compile(
    rf"(?i)\b{_nato_word}(?:[ \t]+{_nato_word}){{1,4}}\b"
)


def normalize_transcript(text: str) -> str:
    """
    Collapses all spelled-out ticker formats into solid uppercase strings
    before the text is printed or sent to the LLM.

      NATO phonetic    Charlie Oscar Sierra        →  COS
                       Alpha Alpha Papa Lima       →  AAPL
                       Tango Sierra Lima Alpha     →  TSLA

      Dot-separated    U.S.A.R.  →  USAR
                       A.A.P.L.  →  AAPL
                       N.V.D.A   →  NVDA  (no trailing dot)

      Hyphen-separated F-C-H-L   →  FCHL
                       T-S-L-A   →  TSLA
    """
    # ── NATO phonetic alphabet ───────────────────────────────────────────────
    def collapse_nato(m: re.Match) -> str:
        words   = m.group(0).lower().split()
        letters = "".join(_NATO.get(w, "") for w in words)
        if 2 <= len(letters) <= 5:
            return letters
        return m.group(0)   # leave unchanged if out of ticker-length range

    text = _NATO_PATTERN.sub(collapse_nato, text)

    # ── Dot-separated: U.S.A.R. or N.V.D.A ─────────────────────────────────
    def collapse_dots(m: re.Match) -> str:
        letters = m.group(0).replace(".", "").upper()
        if 3 <= len(letters) <= 5:   # require 3+ to avoid collapsing "U.S."
            return letters
        return m.group(0)

    text = re.sub(r'(?<!\w)(?:[A-Za-z]\.){2,5}', collapse_dots, text)

    # ── Hyphen-separated: F-C-H-L or T-S-L-A ───────────────────────────────
    def collapse_hyphens(m: re.Match) -> str:
        return m.group(0).replace("-", "").upper()

    text = re.sub(r'(?<!\w)(?:[A-Za-z]-){2,4}[A-Za-z](?!\w)', collapse_hyphens, text)

    return text

# ========================= CONFIG =========================
DEVICE_INDEX   = None          # set to an int to skip the prompt

LLM_MODEL      = "gemma2:2b"
LLM_INTERVAL   = 2.0          # min seconds between LLM calls

WHISPER_MODEL     = "small"   # "tiny" = faster, "small" = more accurate
WHISPER_BEAM_SIZE = 3         # 1 = greedy (fastest); 3 = good balance; 5 = most accurate

SAMPLE_RATE       = 44100
TARGET_SR         = 16000
CHUNK_DURATION    = 4.5       # seconds per Whisper chunk
OVERLAP           = 1.2       # seconds of overlap between chunks
SILENCE_THRESHOLD = 0.009     # RMS below this → skip (checked before resampling)

# Rolling transcript context — how many recent lines to send to the LLM together.
# More lines = better context for ticker detection at the cost of a slightly longer prompt.
TRANSCRIPT_CONTEXT_LINES = 5

# ── Pre-computed constants ────────────────────────────────────────────────────
CHUNK_SAMPLES   = int(TARGET_SR * CHUNK_DURATION)     # 72 000
OVERLAP_SAMPLES = int(TARGET_SR * OVERLAP)            # 19 200
ADVANCE_SAMPLES = CHUNK_SAMPLES - OVERLAP_SAMPLES     # 52 800
READ_FRAMES     = int(SAMPLE_RATE * 0.5)

# 44 100 → 16 000  simplified ratio (GCD=100 → 441/160)
RESAMPLE_UP   = 160
RESAMPLE_DOWN = 441

# ========================= MODEL INIT =========================
print(f"Loading Whisper '{WHISPER_MODEL}' model...")
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("✅ Whisper ready.\n")
print(f"Loading LLM: {LLM_MODEL}...")
print("✅ LLM ready.\n")

# ========================= AUDIO SETUP =========================
p = pyaudio.PyAudio()

print("Available audio input devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev["maxInputChannels"] > 0:
        tag = " ← LOOPBACK (system audio)" if "loopback" in dev["name"].lower() else ""
        print(f"  {i:2d}: {dev['name']}{tag}")

if DEVICE_INDEX is None:
    try:
        choice = input(
            "\nEnter device index (loopback = system audio, mic = AirPods/built-in; "
            "press Enter for default): "
        ).strip()
        DEVICE_INDEX = int(choice) if choice else p.get_default_input_device_info()["index"]
    except Exception:
        DEVICE_INDEX = p.get_default_input_device_info()["index"]

print(f"✅ Using audio device index: {DEVICE_INDEX}\n")

stream = p.open(
    format=pyaudio.paFloat32,
    channels=2,
    rate=SAMPLE_RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=READ_FRAMES,
)

# ========================= SHARED STATE =========================
audio_queue    = Queue(maxsize=12)
llm_queue      = Queue(maxsize=8)
workflow_queue = Queue(maxsize=10)

running = threading.Event()
running.set()

_llm_time_lock = threading.Lock()
_last_llm_time = 0.0


def _get_llm_time() -> float:
    with _llm_time_lock:
        return _last_llm_time


def _set_llm_time(t: float):
    global _last_llm_time
    with _llm_time_lock:
        _last_llm_time = t


# ========================= WORKER: AUDIO CAPTURE =========================
def audio_capture():
    """
    Reads audio → stereo-to-mono → silence gate → resample to 16 kHz → enqueue.
    Silence is gated BEFORE resampling to skip the expensive resample on quiet frames.
    """
    local_buf = np.empty(0, dtype=np.float32)

    while running.is_set():
        try:
            data  = stream.read(READ_FRAMES, exception_on_overflow=False)
            raw   = np.frombuffer(data, dtype=np.float32)
            mono  = raw.reshape(-1, 2).mean(axis=1)

            # Gate on raw mono — skip resampling entirely if silent
            if np.sqrt(np.mean(mono ** 2)) < SILENCE_THRESHOLD:
                continue

            resampled = resample_poly(mono, RESAMPLE_UP, RESAMPLE_DOWN)
            local_buf = np.concatenate((local_buf, resampled))

            while len(local_buf) >= CHUNK_SAMPLES:
                chunk     = local_buf[:CHUNK_SAMPLES].copy()
                local_buf = local_buf[ADVANCE_SAMPLES:]
                try:
                    audio_queue.put_nowait(chunk)
                except Full:
                    pass  # drop rather than stall the audio thread

        except Exception:
            time.sleep(0.05)


# ========================= WORKER: TRANSCRIPTION =========================
def transcription_worker():
    """
    Accumulates audio chunks, runs Whisper, and maintains a rolling buffer of
    recent transcript lines. Every new line triggers an LLM check using the
    last TRANSCRIPT_CONTEXT_LINES lines combined — giving the LLM enough
    context to reliably detect tickers that appear across multiple short chunks.
    """
    local_buf         = np.empty(0, dtype=np.float32)
    transcript_window = deque(maxlen=TRANSCRIPT_CONTEXT_LINES)

    while running.is_set():
        try:
            chunk = audio_queue.get(timeout=1.0)
            if chunk is None:
                break

            local_buf = np.concatenate((local_buf, chunk))

            while len(local_buf) >= CHUNK_SAMPLES:
                whisper_in = local_buf[:CHUNK_SAMPLES]
                local_buf  = local_buf[ADVANCE_SAMPLES:]

                segments, _ = whisper.transcribe(
                    whisper_in,
                    language="en",
                    vad_filter=True,
                    beam_size=WHISPER_BEAM_SIZE,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.45,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                text = normalize_transcript(text)

                if not text or len(text.split()) < 4:
                    continue

                print(f"[{time.strftime('%H:%M:%S')}] 🎤 {text}")

                # Add to rolling window and send combined context to LLM
                transcript_window.append(text)
                if time.time() - _get_llm_time() > LLM_INTERVAL:
                    combined = " ".join(transcript_window)
                    try:
                        llm_queue.put_nowait(combined)
                    except Full:
                        pass

        except Exception:
            continue


# ========================= WORKER: LLM CLASSIFIER =========================
def llm_worker():
    """
    Classifies combined transcript context into buy / sell / watch / NO ACTION.
    Receives the last N transcript lines joined as one string, giving the model
    enough context to detect tickers that were mentioned across several chunks.
    """
    while running.is_set():
        try:
            text = llm_queue.get(timeout=1.0)
            if text is None:
                break

            if time.time() - _get_llm_time() < LLM_INTERVAL:
                continue

            prompt = (
                "You are a strict stock trading signal extractor.\n"
                "Only respond if a clear stock ticker symbol (2-5 uppercase letters "
                "like AAPL, TSLA, UNH, BLD, QXO, CM) is mentioned.\n\n"
                "Reply with EXACTLY one line and nothing else:\n"
                '- "buy TICKER" if buying is mentioned\n'
                '- "sell TICKER" if selling is mentioned\n'
                '- "watch TICKER" if the ticker is mentioned, being tracked, or newsworthy\n'
                '- "NO ACTION" otherwise\n\n'
                f"Text: {text}"
            )

            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": 512, "num_predict": 40},
            )

            _set_llm_time(time.time())

            action = resp["message"]["content"].strip().lower()
            parts  = action.split()

            if len(parts) >= 2 and parts[0] in ("buy", "sell", "watch"):
                verb   = parts[0]
                ticker = parts[1].upper()
                if 2 <= len(ticker) <= 5 and ticker.isalpha():
                    print(f"🤖 Signal: {verb.upper()} {ticker}")
                    try:
                        pyperclip.copy(ticker)
                    except Exception:
                        pass
                    try:
                        workflow_queue.put_nowait((verb, ticker))
                    except Full:
                        pass

        except Exception:
            pass


# ========================= WORKER: WORKFLOW EXECUTOR =========================
def workflow_worker():
    """Runs GUI automation on its own thread — never blocks transcription or LLM."""
    while running.is_set():
        try:
            item = workflow_queue.get(timeout=1.0)
            if item is None:
                break
            verb, ticker = item

            try:
                if verb == "watch":
                    workflow_add_wb(ticker)
                elif verb == "buy":
                    workflow_buy(ticker)
                elif verb == "sell":
                    workflow_sell_all(ticker)
            except Exception as e:
                print(f"⚠️  Workflow error ({verb} {ticker}): {e}")

        except Exception:
            pass


# ========================= START =========================
threads = [
    threading.Thread(target=audio_capture,        daemon=True, name="audio"),
    threading.Thread(target=transcription_worker, daemon=True, name="transcription"),
    threading.Thread(target=llm_worker,           daemon=True, name="llm"),
    threading.Thread(target=workflow_worker,      daemon=True, name="workflow"),
]
for t in threads:
    t.start()

print("🎙️  System running")
print("   Speak a ticker + action  (e.g. 'watch AAPL', 'buy TSLA', 'sell UNH')")
print("   Press Ctrl+C to stop.\n")
wb_watchlist_clear()
print("🗑️  Watchlist cleared.")
wb_watchlist_show()

try:
    while running.is_set():
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\nShutting down...")
    running.clear()
    for q in (audio_queue, llm_queue, workflow_queue):
        try:
            q.put_nowait(None)
        except Full:
            pass

stream.stop_stream()
stream.close()
p.terminate()
print("✅ Stopped cleanly.")

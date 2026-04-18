import pyaudiowpatch as pyaudio
import numpy as np
import time
import sys
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

print("Starting fast transcription...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Whisper ready.\n")

p = pyaudio.PyAudio()
idx = 10

stream = p.open(
    format=pyaudio.paFloat32,
    channels=2,
    rate=44100,
    input=True,
    input_device_index=idx,
    frames_per_buffer=44100
)
print("✅ Transcription running. Play your video.\n")

buffer = []

try:
    while True:
        data = stream.read(44100, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.float32)
        audio = audio.reshape(-1, 2).mean(axis=1)
        audio = resample_poly(audio, 16000, 44100, window=('kaiser', 5.0))

        if np.sqrt(np.mean(audio**2)) < 0.008:
            continue

        buffer.extend(audio)

        if len(buffer) >= 16000 * 2:
            chunk = np.array(buffer[:16000*2], dtype=np.float32)
            buffer = buffer[int(16000*0.8):]

            segments, _ = model.transcribe(
                chunk, language="en", vad_filter=False, beam_size=7, temperature=0.0
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

            if text and len(text) > 10:
                print(f"TEXT:{text}")
                sys.stdout.flush()
except KeyboardInterrupt:
    print("\nTranscription stopped.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
"""Transcribes a recording with the time of every word, on this machine, so voice_align.py can place each line.

    pip install faster-whisper     # in its own environment; the product does not use it
    python transcribe.py <recording> <words.json>

Writes [{"w": word, "start": s, "end": s, "p": probability}]. Nothing leaves the machine; the model downloads once.
"""
import json
import subprocess
import sys

import numpy as np
from faster_whisper import WhisperModel

src, out = sys.argv[1], sys.argv[2]
pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768   # decoded by ffmpeg, as some PyAV builds refuse faster-whisper's open() arguments
segments, _ = WhisperModel("small.en", device="cpu", compute_type="int8").transcribe(audio, word_timestamps=True, beam_size=5, condition_on_previous_text=False)
words = []
for seg in segments:
    print(f"[{seg.start:6.1f}-{seg.end:6.1f}] {seg.text.strip()}", flush=True)
    words += [{"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3), "p": round(w.probability, 3)} for w in seg.words]
with open(out, "w") as f:
    json.dump(words, f)
print(len(words), "words")

"""Cuts a recorded voiceover into the narration lines by the time of each word, for a reading whose pauses are too short to split on.

    python transcribe.py <recording> words.json
    .venv/bin/python tools/video/voice_align.py <recording> words.json [--drop START-END ...]

Each line of voiceover-script.txt is matched, in order, to the transcript's words. Its audio runs from its first matched word
to its last, and each edge moves to the quietest 10 ms near it, so no syllable is clipped. Words between lines, such as a
line number read aloud, are left out. --drop removes a stretch inside a line, such as a false start, with a short
crossfade. Every line is printed beside the words it holds, for checking, and written to data/video/work/voice/<line id>.aiff.
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_split import ORDER, V, VOICE   # noqa: E402

RATE = 16000
FRAME = 0.01        # seconds per energy frame
FADE = 0.015        # seconds of crossfade across a dropped stretch
TARGET_LUFS = -17.0   # spoken-word loudness, a little under the -16 web norm, with the limiter catching peaks at -1.5 dB
PAUSE, KEEP = 0.6, 0.35   # seconds: with --tighten, a pause inside a line longer than PAUSE is cut down to KEEP
SAME = {"storey": "story", "storeys": "stories", "muse": "muse"}


def tokens(text):
    return [SAME.get(t, t) for t in re.findall(r"[a-z]+|\d+", text.lower())]   # numbers digit run by digit run, as 3.1 is heard as 3 and .1


def envelope(path):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    n = int(FRAME * RATE)
    return np.sqrt((x[:len(x) // n * n].reshape(-1, n) ** 2).mean(1))


def edge(db, loud, t, limit, step):
    """The first (step -1) or last (step 1) loud frame of the speech that runs through t, stopping at 30 ms of quiet or at limit."""
    i = int(t / FRAME)
    keep, quiet, j = i, 0, i
    while 0 < j < len(db) - 1 and (j - int(limit / FRAME)) * step < 0 and abs(j - i) < int(0.4 / FRAME):
        j += step
        if db[j] >= loud:
            keep, quiet = j, 0
        else:
            quiet += 1
            if quiet >= 3:
                break
    return keep * FRAME if step < 0 else (keep + 1) * FRAME


def denoise(src, out, span=None, cut_db=30.0):
    """Writes src with its steady room noise taken down by cut_db: a spectral gate learned from span (start, end) in seconds,
    or from the recording's longest stretch with no speech. Speech keeps its level; noise between and under words drops."""
    from scipy.signal import istft, stft
    sr = 48000
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-af", "adeclip", "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768   # clipped peaks repaired before the gate sees them
    if span is None:
        env = envelope(src)
        db = 20 * np.log10(env + 1e-9)
        quiet = np.append(db < float(np.percentile(db, 20)) + 6, False)
        best, run, end = 0, 0, 0
        for i, q in enumerate(quiet):
            run = run + 1 if q else 0
            if run > best:
                best, end = run, i + 1
        span = ((end - best) * FRAME + 0.5, end * FRAME - 0.5)   # keep clear of the speech at either end
        if span[1] - span[0] < 1.0:
            sys.exit("No second of silence to learn the room noise from. Record a few seconds of silence first, or pass --noise START-END.")
    noise = x[int(span[0] * sr):int(span[1] * sr)]
    f, _, N = stft(noise, sr, nperseg=2048, noverlap=1536)
    f, _, X = stft(x, sr, nperseg=2048, noverlap=1536)
    nd = 20 * np.log10(np.abs(N) + 1e-10)
    thresh = nd.mean(1, keepdims=True) + 1.5 * nd.std(1, keepdims=True)   # per frequency: steady noise sits under this
    xd = 20 * np.log10(np.abs(X) + 1e-10)
    mask = 1 / (1 + np.exp(-(xd - thresh) / 2.0))
    k = np.ones((3, 5)) / 15   # smooth the mask over 3 bins and 5 frames, so the gate does not chatter
    from scipy.signal import convolve2d
    mask = convolve2d(mask, k, mode="same", boundary="symm")
    floor = 10 ** (-cut_db / 20)
    _, y = istft(X * (floor + (1 - floor) * mask), sr, nperseg=2048, noverlap=1536)
    y = np.clip(y[:len(x)], -1, 1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-", str(out)], input=(y * 32767).astype(np.int16).tobytes(), check=True)
    return span


def quietest(env, lo, hi):
    a, b = max(int(lo / FRAME), 0), max(int(hi / FRAME), int(lo / FRAME) + 1)
    return (a + int(np.argmin(env[a:b]))) * FRAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("words", type=Path)
    ap.add_argument("--drop", action="append", default=[], help="START-END in seconds of the recording, inside one line")
    ap.add_argument("--tighten", action="store_true", help="shorten long pauses inside each line")
    ap.add_argument("--noise", help="START-END in seconds of room noise with no speech, to learn the noise from; found by itself if left out")
    ap.add_argument("--keep-noise", action="store_true", help="skip the noise reduction")
    ap.add_argument("--start", action="append", default=[], help="LINE=SECONDS to start line LINE (counted from 1) at a chosen point, when the cut needs a person's call")
    a = ap.parse_args()
    spec = json.loads((V / "spec.json").read_text())
    script = [(n, t) for n, k in enumerate(ORDER) for t in tokens(spec["voice"].get(k) or spec["captions"][k])]
    words = json.loads(a.words.read_text())
    heard = [(i, t) for i, w in enumerate(words) for t in tokens(w["w"])]
    matcher = difflib.SequenceMatcher(None, [t for _, t in script], [t for _, t in heard], autojunk=False)
    held = {n: [] for n in range(len(ORDER))}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            held[script[block.a + k][0]].append(heard[block.b + k][0])
    env = envelope(a.recording)
    db = 20 * np.log10(env + 1e-9)
    loud = float(np.percentile(db, 20)) + 12   # the room's floor plus 12 dB counts as speech
    drops = [tuple(map(float, d.split("-"))) for d in a.drop]
    starts = {int(k): float(v) for k, v in (s.split("=") for s in a.start)}
    spans, last = [], -1
    for n, key in enumerate(ORDER):
        if not held[n]:
            sys.exit(f"Line {n + 1} ({key}) matched no word of the recording. Read it again as its own file.")
        first, end = min(held[n]), max(held[n])
        lead = (tokens(spec["voice"].get(key) or spec["captions"][key]) or [""])[0]
        while first - 1 > last and words[first]["start"] - words[first - 1]["end"] < 0.3 and (tokens(words[first - 1]["w"]) or [""])[0][:2] == lead[:2]:
            first -= 1   # a misheard first word, as Muge for Muse, when it starts like the script's; a line number read aloud does not
        while end + 1 < len(words) and words[end + 1]["start"] - words[end]["end"] < 0.25 and not any(end + 1 in held[m] for m in held if m != n):
            end += 1     # the rest of a word the transcript split, as .1. after 3
        if first <= last:
            sys.exit(f"Line {n + 1} ({key}) starts before line {n} ends in the recording, so the lines are out of order.")
        before = words[first - 1]["end"] if first else 0.0
        after = words[end + 1]["start"] if end + 1 < len(words) else words[end]["end"] + 1
        floor_start = spans[-1][2] if spans else 0.0
        if first and words[first]["start"] - before < 0.1:
            # a word left out, such as a line number read aloud, runs into the line: cut in the quiet between them
            start = quietest(env, before - 0.1, words[first]["start"] + 0.15)
        else:
            # Whisper's word times can lag the voice, most at a segment's edge, so the line starts where its speech does
            start = max(edge(db, loud, words[first]["start"], floor_start, -1) - 0.05, floor_start)
        stop = min(edge(db, loud, words[end]["end"], after, 1) + 0.08, after)
        start = starts.get(n + 1, start)
        cuts = [(quietest(env, x - 0.06, x + 0.06), quietest(env, y - 0.06, y + 0.06)) for x, y in drops if start < x < y < stop]
        if a.tighten:   # a pause inside a line longer than PAUSE keeps KEEP seconds, half at each side, so the video stays under its limit
            quiet = db[int(start / FRAME):int(stop / FRAME)] < loud
            run = 0
            for i, q in enumerate(np.append(quiet, False)):
                if q:
                    run += 1
                    continue
                if run * FRAME > PAUSE:
                    qa, qb = start + (i - run) * FRAME, start + i * FRAME
                    if not any(x <= qb and qa <= y for x, y in cuts):
                        cuts.append((qa + KEEP / 2, qb - KEEP / 2))
                run = 0
            cuts.sort()
        spans.append((key, start, stop, cuts))
        mine = [w["w"] for i, w in enumerate(words) if first <= i <= end and not any(x - 0.05 <= w["start"] and w["end"] <= y + 0.05 for x, y in cuts)]
        cover = len(held[n]) / sum(1 for m, _ in script if m == n)
        print(f"{n + 1:2d} {key:13s} {start:7.2f}-{stop:7.2f} {cover:4.0%} {' '.join(mine)}{'   [dropped ' + ', '.join(f'{x:.2f}-{y:.2f}' for x, y in cuts) + ']' if cuts else ''}")
        last = end
    VOICE.mkdir(parents=True, exist_ok=True)
    source = a.recording
    if not a.keep_noise:
        source = VOICE.parent / "recording_clean.wav"
        span = denoise(a.recording, source, tuple(map(float, a.noise.split("-"))) if a.noise else None)
        print(f"Room noise learned from {span[0]:.1f}-{span[1]:.1f} s and taken down 30 dB.")
    # One gain for the whole voice, so every line has the same level and no automatic gain lifts the pauses back up.
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-ss", f"{spans[0][1]:.2f}", "-to", f"{spans[-1][2]:.2f}", "-i", str(source), "-af", "highpass=f=90:poles=2,ebur128",
                            "-f", "null", "-"], capture_output=True, text=True).stderr
    loudness = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", probe)[-1])
    gain = TARGET_LUFS - loudness
    print(f"Voice loudness {loudness:.1f} LUFS, raised {gain:+.1f} dB to {TARGET_LUFS} LUFS.")
    # highpass: the room's rumble sits at 45-150 Hz, under most of a voice. agate: an expander that only acts below -48 dB, in the gaps between words.
    tidy = f"highpass=f=90:poles=2,volume={gain:.2f}dB,agate=threshold=0.004:ratio=2:range=0.1:attack=10:release=250:knee=2.8,alimiter=limit=0.84:level=false,aresample=48000"
    for key, start, stop, cuts in spans:
        edges = [start, *[t for c in cuts for t in c], stop]
        parts = list(zip(edges[::2], edges[1::2]))
        chain = [f"[0:a]atrim={x:.3f}:{y:.3f},asetpts=PTS-STARTPTS[p{i}]" for i, (x, y) in enumerate(parts)]
        last_label = "p0"
        for i in range(1, len(parts)):
            chain.append(f"[{last_label}][p{i}]acrossfade=d={FADE}[x{i}]")
            last_label = f"x{i}"
        chain.append(f"[{last_label}]{tidy}[out]")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-filter_complex", ";".join(chain), "-map", "[out]", "-ac", "2", str(VOICE / f"{key}.aiff")], check=True)
    print(f"Wrote {len(spans)} lines to {VOICE}. Run python3 tools/video/build.py to make the video with this voice.")


if __name__ == "__main__":
    main()

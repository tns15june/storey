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


def quietest(env, lo, hi):
    a, b = max(int(lo / FRAME), 0), max(int(hi / FRAME), int(lo / FRAME) + 1)
    return (a + int(np.argmin(env[a:b]))) * FRAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("words", type=Path)
    ap.add_argument("--drop", action="append", default=[], help="START-END in seconds of the recording, inside one line")
    ap.add_argument("--tighten", action="store_true", help="shorten long pauses inside each line")
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
    tidy = "highpass=f=70,adeclip,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"
    for key, start, stop, cuts in spans:
        edges = [start, *[t for c in cuts for t in c], stop]
        parts = list(zip(edges[::2], edges[1::2]))
        chain = [f"[0:a]atrim={x:.3f}:{y:.3f},asetpts=PTS-STARTPTS[p{i}]" for i, (x, y) in enumerate(parts)]
        last_label = "p0"
        for i in range(1, len(parts)):
            chain.append(f"[{last_label}][p{i}]acrossfade=d={FADE}[x{i}]")
            last_label = f"x{i}"
        chain.append(f"[{last_label}]{tidy}[out]")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.recording), "-filter_complex", ";".join(chain), "-map", "[out]", "-ac", "2", str(VOICE / f"{key}.aiff")], check=True)
    print(f"Wrote {len(spans)} lines to {VOICE}. Run python3 tools/video/build.py to make the video with this voice.")


if __name__ == "__main__":
    main()

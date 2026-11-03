"""Cuts a recorded voiceover into one file per narration line, ready for build.py.

    python3 tools/video/voice_split.py <recording> [line_NN.<ext> ...]

The recording reads the lines of voiceover-script.txt in order, with about 2 seconds of silence between lines. Any format
ffmpeg reads works (m4a from a phone, mp3, wav, aiff). A file named line_NN (NN from 01 to 21) replaces line NN, for a
line read again on its own. Each line is trimmed, set to one loudness and written to data/video/work/voice/<line id>.aiff.

It stops without writing anything when the recording does not split into exactly one piece per line, and says where it
went wrong, because a line placed on the wrong scene is worse than no line.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

V = Path(__file__).resolve().parent
VOICE = V.parent.parent / "data/video/work/voice"
ORDER = ["card_title", "card_problem", "up1", "up2", "up3", "st1", "st2", "st3", "te1", "te2", "te3",
         "fi2", "fi1", "fi3", "ci1", "ci2", "se1", "se2", "le1", "card_numbers", "card_close"]   # the order the video plays them
GAP = 1.2       # seconds of silence that end a line; a breath inside a sentence is shorter
FLOOR = "-35dB"  # quieter than this counts as silence
SHORT = 0.35    # seconds: anything shorter is a click or a breath, not a line
TRIM = f"silenceremove=start_periods=1:start_threshold={FLOOR}:start_silence=0.08"   # drops silence before the first word, and with areverse after the last
WORDS_PER_SECOND = 2.6   # a calm reading pace, only to flag a piece far off its line's length


def spans(path):
    """(start, end) of each stretch of speech in seconds, with a little room kept at each end."""
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout)
    log = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", f"silencedetect=noise={FLOOR}:d={GAP}", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    out, at = [], 0.0
    for n, s in enumerate(starts):
        if s - at > SHORT:
            out.append((max(at - 0.15, 0.0), min(s + 0.15, total)))
        at = ends[n] if n < len(ends) else total
    if total - at > SHORT:
        out.append((max(at - 0.15, 0.0), total))
    return out


def cut(source, start, end, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source),
                    "-af", f"{TRIM},areverse,{TRIM},areverse,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000", "-ac", "2", str(out)], check=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    spec = json.loads((V / "spec.json").read_text())
    text = {k: spec["voice"].get(k) or spec["captions"][k] for k in ORDER}
    recording, extra = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    redo = {}
    for p in extra:
        m = re.fullmatch(r"line_(\d\d)", p.stem)
        if not m or not 1 <= int(m[1]) <= len(ORDER):
            sys.exit(f"{p.name}: a replacement is named line_01 to line_{len(ORDER):02d}.")
        redo[int(m[1])] = p
    pieces = spans(recording)
    print(f"{recording.name}: {len(pieces)} pieces of speech, {len(ORDER)} lines expected")
    for n, (key, (a, b)) in enumerate(zip(ORDER, pieces), 1):
        want = len(text[key].split()) / WORDS_PER_SECOND
        flag = "" if 0.5 <= (b - a) / want <= 2.0 else "   <- far from this line's length"
        print(f"  {n:2d} {key:13s} {b - a:5.1f} s (about {want:4.1f} s expected){flag}")
    if len(pieces) != len(ORDER):
        sys.exit(f"Not written: the recording split into {len(pieces)} pieces, not {len(ORDER)}. A line that stopped for over {GAP} s "
                 "splits in two, and a line read twice counts twice. Leave a clear pause between lines only, or read the faulty "
                 "lines again as line_NN files.")
    VOICE.mkdir(parents=True, exist_ok=True)
    for n, (key, (a, b)) in enumerate(zip(ORDER, pieces), 1):
        if n in redo:
            r = spans(redo[n])
            cut(redo[n], r[0][0], r[-1][1], VOICE / f"{key}.aiff")
        else:
            cut(recording, a, b, VOICE / f"{key}.aiff")
    print(f"Wrote {len(ORDER)} lines to {VOICE}. Run python3 tools/video/build.py to make the video with this voice.")


if __name__ == "__main__":
    main()

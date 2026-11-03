"""Cuts the takes, frames them on the dark page with their captions and narration, adds the cards, and joins everything into storey.mp4.

    python3 tools/video/build.py [--silent]

Reads data/video/work/clips (rec.mjs takes) and data/video/work/cards (cards.mjs), writes data/video/work/storey.mp4. data/ is not in git.
Narration: data/video/work/voice/<line id>.aiff for each caption and card in spec.json. A missing line is spoken by macOS `say`
(Samantha, 182 words a minute), so a recorded voice replaces it by putting a file of the same name there.
A segment whose narration runs past its footage holds its last frame until the line ends.
"""
import json
import subprocess
import sys
from pathlib import Path

V = Path(__file__).resolve().parent
WORK = V.parent.parent / "data/video/work"
CLIPS, CARDS, SEGS, VOICE = WORK / "clips", WORK / "cards", WORK / "segs", WORK / "voice"
SEGS.mkdir(parents=True, exist_ok=True)
VOICE.mkdir(parents=True, exist_ok=True)
SILENT = "--silent" in sys.argv
SPEC = json.loads((V / "spec.json").read_text())
for line, text in {**SPEC["captions"], **SPEC["voice"]}.items():
    if not SILENT and not (VOICE / f"{line}.aiff").exists():
        subprocess.run(["say", "-v", "Samantha", "-r", "182", "-o", str(VOICE / f"{line}.aiff"), text], check=True)
LEAD = 1.3   # seconds of recording before each take's first mark
FPS = 30
TAIL = 0.5   # seconds after a line before the next segment
ENC = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]


def seconds(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout)


def marks(take):
    return {m["name"]: m["s"] + LEAD for m in json.loads((CLIPS / f"{take}.json").read_text())["marks"]}


def audio(lines, length, first_input):
    """(inputs, filter lines, output label) placing each (voice id, start) at its start, over silence of `length`."""
    inputs, chain, labels = [], [], []
    for n, (vid, at) in enumerate(lines):
        if SILENT:
            break
        inputs += ["-i", str(VOICE / f"{vid}.aiff")]
        chain.append(f"[{first_input + n}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={int(at * 1000)}|{int(at * 1000)}[a{n}]")
        labels.append(f"[a{n}]")
    chain.append(f"anullsrc=channel_layout=stereo:sample_rate=48000,atrim=0:{length:.3f}[sil]")
    chain.append(f"[sil]{''.join(labels)}amix=inputs={len(labels) + 1}:duration=first:normalize=0,atrim=0:{length:.3f}[aout]")
    return inputs, chain


def out_name(tag):
    return SEGS / f"{len(list(SEGS.glob('*.mp4'))):02d}_{tag}.mp4"


def card(name, minimum):
    vid = f"card_{name}"
    length = max(minimum, seconds(VOICE / f"{vid}.aiff") + 0.6 + TAIL) if not SILENT else minimum
    out = out_name(f"card_{name}")
    ain, achain = audio([(vid, 0.4)], length, 1)
    fade = f"fade=t=in:st=0:d=0.4,fade=t=out:st={length - 0.4:.2f}:d=0.4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", f"{length:.3f}", "-i", str(CARDS / f"card_{name}.png"), *ain,
                    "-filter_complex", ";".join([f"[0:v]fps={FPS},format=yuv420p,{fade}[vout]", *achain]),
                    "-map", "[vout]", "-map", "[aout]", "-t", f"{length:.3f}", *ENC, str(out)], check=True)
    return out


def clip(take, start, end, speed=1.0, captions=(), fade_in=False, fade_out=False):
    """`captions`: (id, from, to) in seconds of this segment; `to` None means to the end. Each caption's narration starts at `from`."""
    footage = (end - start) / speed
    need = max((a + seconds(VOICE / f"{cap}.aiff") + TAIL for cap, a, _ in captions), default=0) if not SILENT else 0
    length = max(footage, need)
    out = out_name(f"{take}_{int(start)}")
    inputs = ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(CLIPS / f"{take}.webm")]
    hold = f",tpad=stop_mode=clone:stop_duration={length - footage + 0.1:.3f}" if length > footage else ""
    chain = [f"[0:v]setpts=(PTS-STARTPTS)/{speed},fps={FPS}{hold},scale=1600:900:flags=lanczos,pad=1920:1080:160:24:color=0x12161d,"
             f"drawbox=x=158:y=22:w=1604:h=904:color=0x333b49:t=2[v0]"]
    last = "v0"
    for n, (cap, a, b) in enumerate(captions, 1):
        inputs += ["-loop", "1", "-t", f"{length:.3f}", "-i", str(CARDS / f"cap_{cap}.png")]
        b = length if b is None else b
        chain.append(f"[{last}][{n}:v]overlay=0:0:enable='between(t,{a:.2f},{b:.2f})'[v{n}]")
        last = f"v{n}"
    fades = ",".join(f for f in (fade_in and "fade=t=in:st=0:d=0.4", fade_out and f"fade=t=out:st={length - 0.4:.2f}:d=0.4") if f)
    chain.append(f"[{last}]format=yuv420p{',' + fades if fades else ''}[vout]")
    ain, achain = audio([(cap, a) for cap, a, _ in captions], length, 1 + len(captions))
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, *ain, "-filter_complex", ";".join(chain + achain),
                    "-map", "[vout]", "-map", "[aout]", "-t", f"{length:.3f}", *ENC, str(out)], check=True)
    return out


for old in SEGS.glob("*.mp4"):
    old.unlink()
parts = [card("title", 5.5), card("problem", 9.5)]

m = marks("upload")
a, b = m["loaded"] + 0.3, m["started"] + 1.0
parts.append(clip("upload", a, b, captions=[("up1", 0, None)], fade_in=True))
parts.append(clip("upload", b, m["done"], speed=16, captions=[("up2", 0, None)]))
parts.append(clip("upload", m["done"], m["end"], captions=[("up3", 0, None)]))

m = marks("stairs")
parts.append(clip("stairs", m["loaded"] + 0.4, m["cell"], speed=1.8, captions=[("st1", 0, None)]))
parts.append(clip("stairs", m["cell"], m["found"], captions=[("st2", 0, None)]))
parts.append(clip("stairs", m["found"], m["end"], captions=[("st3", 0, None)]))

m = marks("test")
parts.append(clip("test", m["loaded"] + 0.4, m["removed"] + 0.8, captions=[("te1", 0, None)]))
parts.append(clip("test", m["removed"] + 0.8, m["done"], speed=20, captions=[("te2", 0, None)]))
parts.append(clip("test", m["done"], m["end"] - 1.5, captions=[("te3", 0, None)]))

m = marks("findings")
s = m["loaded"] + 0.4
parts.append(clip("findings", s, m["end"], captions=[("fi2", 0, m["second"] - s), ("fi1", m["second"] - s, m["notchecked"] - s), ("fi3", m["notchecked"] - s, None)]))

m = marks("city")
s = m["loaded"] + 0.4
parts.append(clip("city", s, m["seattle"] + 6.2, captions=[("ci1", 0, m["seattle"] - s), ("ci2", m["seattle"] - s, None)]))

m = marks("second")
s = m["loaded"] + 0.4
parts.append(clip("second", s, m["end"] - 1.5, captions=[("se1", 0, 4.8), ("se2", 4.8, None)]))

m = marks("letter")
parts.append(clip("letter", m["loaded"] + 0.4, m["end"], captions=[("le1", 0, None)], fade_out=True))

parts.append(card("numbers", 10.5))
parts.append(card("close", 5.0))

listing = WORK / "concat.txt"
listing.write_text("".join(f"file '{p}'\n" for p in parts))
final = WORK / ("storey_silent.mp4" if SILENT else "storey.mp4")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
print(final.name, round(seconds(final), 1), "s,", len(parts), "segments")

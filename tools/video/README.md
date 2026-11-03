# How the demo video was made

The video at `app/web/video/storey.mp4` (2:33) is a screen recording of the real app, cut and captioned by these scripts. Nothing in it is mocked: the upload and the stair removal are live runs on Meta's models, shown sped up.

1. Start the server on the reviews to show, with a known key:
   `STOREY_DATA=<reviews folder> STOREY_ACCESS_KEY=<key> .venv/bin/python -m storey --port 8811`
2. Record each take in Chrome, one per shot (Playwright, `npm install playwright` in this folder):
   `STOREY_ACCESS_KEY=<key> node rec.mjs <shot> ../../data/video/work/clips reviews.json`
   Shots: `upload`, `stairs`, `test`, `findings`, `city`, `second`, `letter`. `reviews.json` maps `stel`, `capitol`, `mixed` and `second` to review ids on that server. `upload` and `test` start real runs and spend about 0.40 and 0.35 USD.
3. Render the title cards and caption bands from `spec.json`:
   `node cards.mjs spec.json ../../data/video/work/cards`
4. Cut, caption, narrate and join: `python3 build.py` writes `data/video/work/storey.mp4`; `python3 build.py --silent` writes a cut with no voice.

Narration is macOS `say` (Samantha). To use a recorded voice, read the lines of `voiceover-script.txt` in one recording, with a pause between lines, then:

    python3 tools/video/voice_split.py <recording> [line_NN.m4a ...]
    python3 tools/video/build.py

`voice_split.py` cuts the recording into one file per line, evens the loudness, and stops without writing if the recording does not split into exactly 21 lines. A `line_NN` file replaces line NN.

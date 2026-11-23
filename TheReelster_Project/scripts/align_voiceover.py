#!/usr/bin/env python3
"""Time the approved script against the supplied voiceover - fully local (no speech-to-text service).

1. Convert the supplied MP3 -> assets/audio/voiceover_01.wav (48 kHz stereo) and a 16 kHz mono copy for analysis.
2. Forced-align the EXACT approved script text to the audio with PocketSphinx (acoustic model ships in the
   wheel, nothing is downloaded) -> start/end time of every word.
3. Cross-check against independent FFmpeg silence detection.
4. Write assets/audio/voiceover_timing.json (seconds + 30 fps frames, per word and per line) and print a report.

The script text below is the approved narration; it is never edited by this tool.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pocketsphinx as ps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config

cfg = load_config()
FPS = cfg.fps
SRC = cfg.path("assets/audio/source/voiceover_elevenlabs_original.mp3")
WAV = cfg.path("assets/audio/voiceover_01.wav")
OUT = cfg.path("assets/audio/voiceover_timing.json")

# (scene number, narration) - APPROVED SCRIPT, verbatim. Scene 12 has no narration.
SCRIPT = [
    (1, "Your mind has weaknesses you probably don't even know about."),
    (2, "First, social exclusion: being ignored can threaten our sense of belonging."),
    (3, "That feeling can hurt, even when no one says a word."),
    (4, "Next, mirroring: people often unconsciously copy gestures and expressions."),
    (5, "This Chameleon Effect can help interactions feel more natural."),
    (6, "Then there's scarcity: limited availability can make things seem more desirable."),
    (7, "That's why countdowns and 'only a few left' can influence decisions."),
    (8, "The foot-in-the-door effect starts with a small request."),
    (9, "Agreeing once can make a larger request easier to accept."),
    (10, "But threaten someone's freedom, and reactance may make them push back."),
    (11, "Sometimes, controlling your choices starts by making you feel you have none."),
]
EXTRA_PRONUNCIATIONS = {"reactance": "R IY AE K T AH N S"}   # not in the CMU dictionary


def tokens(text: str) -> list[str]:
    t = re.sub(r"[^\w' -]", "", text.lower()).replace("-", " ")
    return [w.strip("'") for w in t.split() if w.strip("'")]


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    if not SRC.is_file():
        sys.exit(f"MISSING voiceover: {SRC}")
    # 1. conversions
    run(["ffmpeg", "-hide_banner", "-y", "-i", str(SRC), "-ar", str(cfg.audio["sample_rate"]), "-ac", "2",
         "-c:a", "pcm_s16le", str(WAV)])
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(SRC), "-ar", "16000", "-ac", "1",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    total = len(raw) / 2 / 16000

    # 2. forced alignment of the full script
    words = [(sc, w) for sc, line in SCRIPT for w in tokens(line)]
    dec = ps.Decoder(loglevel="ERROR")
    for w, ph in EXTRA_PRONUNCIATIONS.items():
        dec.add_word(w, ph, True)
    missing = [w for _, w in words if dec.lookup_word(w) is None]
    if missing:
        sys.exit(f"words not in dictionary: {missing}")
    dec.set_align_text(" ".join(w for _, w in words))
    dec.start_utt()
    dec.process_raw(raw, False, True)
    dec.end_utt()
    aligned = [(re.sub(r"\(\d+\)$", "", sg.word), sg.start_frame / 100.0, (sg.end_frame + 1) / 100.0)     # PocketSphinx frames are 10 ms
               for sg in dec.seg() if sg.word not in ("<sil>", "(NULL)", "<s>", "</s>")]
    if len(aligned) != len(words):
        sys.exit(f"alignment returned {len(aligned)} words, expected {len(words)}")

    # 3. independent check: FFmpeg silence detection
    sd = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(SRC), "-af", "silencedetect=noise=-38dB:d=0.18",
                         "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", sd)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", sd)]

    # 4. assemble per word / per line
    lines, i = [], 0
    for sc, text in SCRIPT:
        n = len(tokens(text))
        ws = [{"word": w, "start": round(s, 3), "end": round(e, 3),
               "start_frame": round(s * FPS), "end_frame": round(e * FPS)} for w, s, e in aligned[i:i + n]]
        i += n
        lines.append({"scene": sc, "text": text, "start": ws[0]["start"], "end": ws[-1]["end"],
                      "start_frame": ws[0]["start_frame"], "end_frame": ws[-1]["end_frame"],
                      "window": [(sc - 1) * 5.0, sc * 5.0], "words": ws})
    data = {"voiceover": str(WAV.relative_to(cfg.root)), "duration": round(total, 3), "fps": FPS,
            "method": "PocketSphinx forced alignment of the approved script (local); cross-checked with FFmpeg silencedetect",
            "silences": [{"start": s, "end": e} for s, e in zip(starts, ends)], "lines": lines}
    OUT.write_text(json.dumps(data, indent=2))

    # report
    print(f"voiceover: {total:.2f}s -> {WAV.name}\n")
    print(f"{'sc':>2} {'start':>6} {'end':>6} {'len':>5} {'scene window':>13}  {'fits 5s?':<9} text")
    for ln in lines:
        w0, w1 = ln["window"]
        fits = "ok" if ln["start"] >= w0 - 0.05 and ln["end"] <= w1 + 0.05 else ("OVERRUNS" if ln["end"] > w1 + 0.05 else "early")
        print(f"{ln['scene']:>2} {ln['start']:6.2f} {ln['end']:6.2f} {ln['end'] - ln['start']:5.2f} {w0:5.0f}-{w1:<5.0f}s  {fits:<9} {ln['text'][:48]}")
    return data


if __name__ == "__main__":
    main()

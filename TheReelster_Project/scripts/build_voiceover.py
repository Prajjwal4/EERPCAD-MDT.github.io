#!/usr/bin/env python3
"""Assemble the final voiceover from the per-line ElevenLabs takes and time every word. Local only.

Per line: trim silence -> force-align the approved text to that clip (PocketSphinx, local) -> match level.
Then lay the lines on the 5 s scene grid and write
    assets/audio/voiceover_01.wav          60.000 s, 48 kHz stereo, -16 LUFS, peak <= -1.5 dBFS
    assets/audio/voiceover_timing.json     start/end of every line and word (seconds and 30 fps frames)

The spoken words are never edited or time-stretched; only the pauses BETWEEN lines are chosen here.
Scheduling rule: line k wants to start LEAD s into scene k (5 s windows); lines never overlap
(MIN_GAP between them); if the chain would push the last line past END_CAP (scene 12 is voice-free),
late lines are pulled earlier just enough.
"""
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pocketsphinx as ps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import measure_loudness, normalize_voiceover
from config import load_config

cfg = load_config()
SR, FPS = cfg.audio["sample_rate"], cfg.fps
TAKES = cfg.path("assets/audio/takes")
OUT_WAV = cfg.path("assets/audio/voiceover_01.wav")
OUT_JSON = cfg.path("assets/audio/voiceover_timing.json")
LEAD, MIN_GAP, END_CAP, TOTAL = 0.15, 0.30, 54.9, 60.0

# APPROVED SCRIPT (verbatim) and the take chosen for each line.
LINES = [
    (1, "Your mind has weaknesses you probably don't even know about.", "s01_take2.mp3"),
    (2, "First, social exclusion: being ignored can threaten our sense of belonging.", "s02_take2.mp3"),
    (3, "That feeling can hurt, even when no one says a word.", "s03_take2.mp3"),
    (4, "Next, mirroring: people often unconsciously copy gestures and expressions.", "s04_take2.mp3"),
    (5, "This Chameleon Effect can help interactions feel more natural.", "s05_take1.mp3"),
    (6, "Then there's scarcity: limited availability can make things seem more desirable.", "s06_take2.mp3"),
    (7, "That's why countdowns and 'only a few left' can influence decisions.", "s07_take1.mp3"),
    (8, "The foot-in-the-door effect starts with a small request.", "s08_take1.mp3"),
    (9, "Agreeing once can make a larger request easier to accept.", "s09_take2.mp3"),
    (10, "But threaten someone's freedom, and reactance may make them push back.", "s10_take2.mp3"),
    (11, "Sometimes, controlling your choices starts by making you feel you have none.", "s11_take3.mp3"),
]
PRON = {"reactance": "R IY AE K T AH N S"}


def tokens(text):
    t = re.sub(r"[^\w' -]", "", text.lower()).replace("-", " ")
    return [w.strip("'") for w in t.split() if w.strip("'")]


def decode(path, rate):
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-ar", str(rate), "-ac", "1",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def trim(x, rate, thresh_db=-45.0, pad=0.03):
    """Cut leading/trailing silence, keep `pad` seconds. Returns (clip, seconds removed from the front)."""
    win = int(rate * 0.01)
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, "same"))
    idx = np.where(env > 10 ** (thresh_db / 20))[0]
    a, b = max(0, idx[0] - int(pad * rate)), min(len(x), idx[-1] + int(pad * rate))
    return x[a:b], a / rate


def align(clip16k, text):
    dec = ps.Decoder(loglevel="ERROR")
    for w, ph in PRON.items():
        dec.add_word(w, ph, True)
    words = tokens(text)
    miss = [w for w in words if dec.lookup_word(w) is None]
    if miss:
        sys.exit(f"not in dictionary: {miss}")
    dec.set_align_text(" ".join(words))
    dec.start_utt()
    dec.process_raw((np.clip(clip16k, -1, 1) * 32767).astype("<i2").tobytes(), False, True)
    dec.end_utt()
    segs = [(re.sub(r"\(\d+\)$", "", s.word), s.start_frame / 100.0, (s.end_frame + 1) / 100.0)
            for s in dec.seg() if s.word not in ("<sil>", "(NULL)", "<s>", "</s>")]
    if [s[0] for s in segs] != words:
        sys.exit(f"alignment mismatch for {text!r}")
    return segs


def lufs(x, rate):
    tmp = TAKES / "_tmp_lufs.wav"
    write_wav(tmp, x, rate)
    v = measure_loudness(tmp)["I"]
    tmp.unlink()
    return v


def write_wav(path, mono_or_stereo, rate):
    a = np.asarray(mono_or_stereo, np.float32)
    if a.ndim == 1:
        a = np.stack([a, a], 1)
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())


def main():
    clips = []
    for sc, text, fn in LINES:
        p = TAKES / fn
        if not p.is_file():
            sys.exit(f"MISSING take: {p}")
        x48, x16 = decode(p, SR), decode(p, 16000)
        c48, lead48 = trim(x48, SR)
        c16, _ = trim(x16, 16000)
        words = align(c16, text)
        clips.append({"scene": sc, "text": text, "take": fn, "x": c48, "dur": len(c48) / SR, "words": words,
                      "lufs": lufs(c48, SR)})
    med = float(np.median([c["lufs"] for c in clips]))
    for c in clips:                                           # level-match lines (max +/-3 dB)
        g = float(np.clip(med - c["lufs"], -3, 3))
        c["gain_db"] = round(g, 2)
        c["x"] = c["x"] * 10 ** (g / 20)

    # ---- schedule on the 5 s grid
    fwd, prev_end = [], -9.0
    for c in clips:
        s = max(5.0 * (c["scene"] - 1) + LEAD, prev_end + MIN_GAP)
        fwd.append(s); prev_end = s + c["dur"]
    latest, nxt = [0.0] * len(clips), END_CAP
    for i in range(len(clips) - 1, -1, -1):
        latest[i] = nxt - clips[i]["dur"]
        nxt = latest[i] - MIN_GAP
    starts = [min(f, l) for f, l in zip(fwd, latest)]
    for i in range(1, len(clips)):
        assert starts[i] >= starts[i - 1] + clips[i - 1]["dur"] + MIN_GAP - 1e-6, "lines overlap"
    assert starts[0] >= 0 and starts[-1] + clips[-1]["dur"] <= END_CAP + 1e-6

    # ---- mix onto the 60 s track
    track = np.zeros(int(TOTAL * SR), np.float32)
    fade = int(0.008 * SR)
    for c, s in zip(clips, starts):
        x = c["x"].copy()
        x[:fade] *= np.linspace(0, 1, fade); x[-fade:] *= np.linspace(1, 0, fade)
        i0 = int(round(s * SR))
        track[i0:i0 + len(x)] += x
    raw = TAKES / "_assembled_raw.wav"
    write_wav(raw, track, SR)
    info = normalize_voiceover(raw, OUT_WAV, cfg)             # two-pass loudnorm: -16 LUFS, TP <= -1.5
    raw.unlink()

    # ---- timing JSON (global times)
    data = {"voiceover": str(OUT_WAV.relative_to(cfg.root)), "duration": TOTAL, "fps": FPS, "voice": "Michael C. Vincent - Suspenseful Storyteller (ElevenLabs eleven_v4)",
            "method": "per-line ElevenLabs takes; per-clip PocketSphinx forced alignment (local); lines placed on the 5 s scene grid",
            "loudness": info["after"], "lines": []}
    for c, s in zip(clips, starts):
        ws = [{"word": w, "start": round(s + a, 3), "end": round(s + b, 3),
               "start_frame": round((s + a) * FPS), "end_frame": round((s + b) * FPS)} for w, a, b in c["words"]]
        data["lines"].append({"scene": c["scene"], "text": c["text"], "take": c["take"], "gain_db": c["gain_db"],
                              "start": round(s, 3), "end": round(s + c["dur"], 3),
                              "start_frame": round(s * FPS), "end_frame": round((s + c["dur"]) * FPS),
                              "window": [(c["scene"] - 1) * 5.0, c["scene"] * 5.0], "words": ws})
    OUT_JSON.write_text(json.dumps(data, indent=2))

    print(f"{'sc':>2} {'start':>6} {'end':>6} {'len':>5} {'vs scene start':>14} {'spill past scene':>17} {'gain':>6}  take")
    for l in data["lines"]:
        w0, w1 = l["window"]
        print(f"{l['scene']:>2} {l['start']:6.2f} {l['end']:6.2f} {l['end'] - l['start']:5.2f} {l['start'] - w0:+14.2f} {max(0, l['end'] - w1):17.2f} {l['gain_db']:+6.1f}  {l['take']}")
    a = info["after"]
    print(f"\nvoiceover_01.wav: {TOTAL:.1f}s  loudness {a['I']} LUFS  true peak {a['TP']} dBTP  sample peak {a['sample_peak']} dBFS")
    print(f"last narration ends {data['lines'][-1]['end']:.2f}s  (scene 12 voice-free from 55.00s)")


if __name__ == "__main__":
    main()

"""Procedurally synthesise the original music bed and SFX (no samples, no copyrighted material)."""
import sys, wave
from pathlib import Path
import numpy as np

SR = 48000
rng = np.random.default_rng(1)  # deterministic

def write(path, x):
    x = np.clip(x, -1, 1)
    pcm = (x * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())

def t(d): return np.arange(int(SR * d)) / SR
def lp(x, fc):  # one-pole low-pass
    a = np.exp(-2 * np.pi * fc / SR); y = np.empty_like(x); s = 0.0
    for i, v in enumerate(x): s = (1 - a) * v + a * s; y[i] = s
    return y
def stereo(l, r=None): return np.stack([l, l if r is None else r], 1)
def norm(x, peak): return x / (np.abs(x).max() + 1e-9) * peak

def music(d=60.0):
    tt = t(d); out = np.zeros((len(tt), 2))
    # A-minor drone: A1 + E2 + C3 with slow detuned pairs for width
    for f, g in [(55, .5), (82.41, .3), (130.81, .18), (65.41, .15)]:
        for side, det in ((0, 0.15), (1, -0.15)):
            lfo = 1 + .3 * np.sin(2 * np.pi * (.05 + .02 * side) * tt + rng.uniform(0, 6))
            out[:, side] += g * np.sin(2 * np.pi * (f + det) * tt) * lfo
    # filtered noise pad that swells every 12 s
    n = lp(rng.standard_normal(len(tt)), 400)
    swell = .5 + .5 * np.sin(2 * np.pi * tt / 12 - 1.5)
    out += stereo(n * swell * 3.0, np.roll(n, 2400) * swell * 3.0)
    # sparse low pulse every 4 s
    pulse = np.zeros(len(tt))
    for s in np.arange(2, d, 4.0):
        i = int(s * SR); k = t(1.5)[: len(pulse) - i]
        pulse[i:i + len(k)] += np.sin(2 * np.pi * 41 * k) * np.exp(-3 * k)
    out += stereo(pulse * .6)
    out = norm(out, .8)
    fi, fo = int(1.5 * SR), int(2.0 * SR)
    out[:fi] *= np.linspace(0, 1, fi)[:, None]; out[-fo:] *= np.linspace(1, 0, fo)[:, None]
    return out

def impact(d=2.0):
    k = t(d); f = 38 + 90 * np.exp(-12 * k)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-2.8 * k)
    hit = lp(rng.standard_normal(len(k)), 3000) * np.exp(-25 * k) * .6
    return stereo(norm(body + hit, .9))

def whoosh(d=2.0):
    k = t(d); env = np.sin(np.pi * np.clip(k / 1.2, 0, 1)) ** 2 * (k < 1.2)
    n = rng.standard_normal(len(k)); y = np.zeros_like(n)
    for fc_lo, fc_hi in [(300, 4000)]:
        a = lp(n, 4000) - lp(n, 300)
    sweep = a * env
    return stereo(norm(sweep, .7), norm(np.roll(sweep, 300), .7))

def heartbeat(d=2.0):
    k = t(d); y = np.zeros(len(k))
    for s in (0.10, 0.34, 1.10, 1.34):  # lub-dub x2 (~55 bpm)
        i = int(s * SR); kk = t(.35)[: len(y) - i]
        y[i:i + len(kk)] += np.sin(2 * np.pi * 52 * kk) * np.exp(-14 * kk) * (1.0 if (s % 1) < .2 else .7)
    return stereo(norm(y, .9))

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "assets"); out.mkdir(exist_ok=True)
    write(out / "bg_music.wav", music()); write(out / "sfx_01.wav", impact())
    write(out / "sfx_02.wav", whoosh()); write(out / "sfx_03.wav", heartbeat())
    print("wrote bg_music.wav sfx_01..03.wav")

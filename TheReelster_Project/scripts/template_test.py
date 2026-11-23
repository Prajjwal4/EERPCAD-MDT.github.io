#!/usr/bin/env python3
"""Framework self-test. Renders renders/template_test.mp4 (2 s, 60 frames) from a TEMPORARY
test image, then verifies it with ffprobe and extracts frames to qa/. Not a final scene."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_renderer import CaptionGroup
from config import load_config
from scene_renderer import Camera, CaptionEvent, FFmpegWriter, Scene, SceneRenderer

cfg = load_config()
R = SceneRenderer(cfg)
OUT = cfg.path("renders/template_test.mp4")
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def make_test_image() -> Image.Image:
    """Landscape 2400x1500 (wrong aspect on purpose -> exercises cover-crop). Corner markers
    prove edges are filled; horizontal ramp + circle give the grade/zoom something to act on."""
    w, h = 2400, 1500
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    arr = np.concatenate([0.15 + 0.7 * x * np.ones_like(y), 0.25 + 0.5 * y * np.ones_like(x), 0.6 - 0.4 * x * np.ones_like(y)], 2)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for gx in range(0, w, 200):
        d.line([(gx, 0), (gx, h)], fill=(255, 255, 255), width=2)
    for gy in range(0, h, 200):
        d.line([(0, gy), (w, gy)], fill=(255, 255, 255), width=2)
    d.ellipse([w / 2 - 350, h / 2 - 350, w / 2 + 350, h / 2 + 350], outline=(255, 255, 255), width=14)
    return img


# --- rule checks (should fail loudly) --------------------------------------
check("caption >7 words rejected", raises(lambda: R.captions.build(CaptionGroup("one two three four five six seven eight")), ValueError))
check("non-150-frame scene rejected outside test mode", raises(lambda: list(R.iter_frames(Scene(image_obj=make_test_image(), frames=151))), ValueError))
check("missing asset raises (no substitution)", raises(lambda: R.load_image(Scene(image=Path("assets/visuals/visual_01.png"))), FileNotFoundError))

# --- safe margins & highlighting ------------------------------------------
for kind, pos, text, hl in [
    ("caption", "lower", "Your brain copies what it sees", ("copies",)),
    ("caption", "center", "Scarcity makes things feel urgent now", ("Scarcity",)),
    ("headline", "upper", "Psychological Tricks Everywhere", ("Psychological",)),
    ("headline", "upper", "MIRRORING", ()),
]:
    ov = R.captions.build(CaptionGroup(text, hl), kind, pos)
    l, t, r, b = cfg.safe_rect
    ok = ov.ink_bbox[0] >= l and ov.ink_bbox[1] >= t and ov.ink_bbox[2] <= r and ov.ink_bbox[3] <= b and len(ov.lines) <= 3
    check(f"{kind}/{pos} inside safe rect", ok, f"font={ov.font_px}px lines={ov.lines} ink={ov.ink_bbox}")
red = cfg.rgb("accent_red")
ov = R.captions.build(CaptionGroup("Your brain copies what it sees", ("copies",)))
px = (ov.rgb * 255).round().astype(int).transpose(1, 2, 0)[ov.alpha > 0.95].reshape(-1, 3)
has_red = bool(((np.abs(px - np.array(red)).sum(1)) < 6).any())
has_white = bool(((np.abs(px - 255).sum(1)) < 6).any())
check("highlight red + normal white both present", has_red and has_white)

# --- render the 2 s test ---------------------------------------------------
t0 = time.time()
scene = Scene(
    image_obj=make_test_image(),
    camera=Camera(zoom_start=1.0, zoom_end=1.18, pan_start=(-0.6, 0.0), pan_end=(0.6, 0.0)),
    frames=60, fade_in=9, fade_out=9,
    captions=[
        CaptionEvent(CaptionGroup("MIRRORING", ()), 6, 54, kind="headline", position="upper"),
        CaptionEvent(CaptionGroup("Your brain copies what it sees", ("copies",)), 10, 34, position="lower"),
        CaptionEvent(CaptionGroup("People trust the mirror effect", ("mirror effect",)), 34, 54, position="lower"),
    ],
)
with FFmpegWriter(OUT, cfg, preset="medium") as w:
    n = R.render_scene(scene, w, test_mode=True)
print(f"rendered {n} frames in {time.time() - t0:.1f}s -> {OUT}")

# --- ffprobe verification ---------------------------------------------------
pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,duration,color_space",
                                "-of", "json", str(OUT)], capture_output=True, text=True, check=True).stdout)["streams"][0]
print(json.dumps(pr, indent=2))
check("dimensions 1080x1920", (pr["width"], pr["height"]) == (1080, 1920))
check("frame rate 30/1", pr["r_frame_rate"] == "30/1" and pr["avg_frame_rate"] == "30/1")
check("frame count 60", int(pr["nb_read_frames"]) == 60 and n == 60)
check("h264 / yuv420p", pr["codec_name"] == "h264" and pr["pix_fmt"] == "yuv420p")

# --- extract frames ---------------------------------------------------------
qa = cfg.path("qa")
for idx, label in [(0, "first"), (24, "mid"), (45, "late"), (59, "last")]:
    dst = qa / f"template_test_frame_{label}.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(OUT), "-vf", f"select=eq(n\\,{idx})",
                    "-frames:v", "1", str(dst)], check=True)
    im = Image.open(dst)
    check(f"frame {idx} extracted at 1080x1920", im.size == (1080, 1920), f"mean={np.asarray(im).mean():.1f}")
f0, fl = np.asarray(Image.open(qa / "template_test_frame_first.png")).mean(), np.asarray(Image.open(qa / "template_test_frame_last.png")).mean()
fm = np.asarray(Image.open(qa / "template_test_frame_mid.png")).mean()
check("fades: first & last frames darker than mid", f0 < fm * 0.2 and fl < fm * 0.2, f"first={f0:.1f} mid={fm:.1f} last={fl:.1f}")

# --- audio_utils smoke test on temp synthetic audio (not project assets) ------
from audio_utils import measure_loudness, mix_final_audio, normalize_voiceover, probe
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    def ff(*a): subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *a], check=True)
    ff("-f", "lavfi", "-i", "sine=f=180:d=20:sample_rate=44100", "-af", "tremolo=f=3:d=0.8,volume=0.9", str(td / "vo_raw.wav"))
    ff("-f", "lavfi", "-i", "anoisesrc=d=60:c=pink:a=0.3:r=48000", "-ac", "2", str(td / "music.wav"))
    ff("-f", "lavfi", "-i", "sine=f=60:d=2:sample_rate=48000", "-af", "afade=t=out:st=0.2:d=1.8", str(td / "sfx.wav"))
    nv = normalize_voiceover(td / "vo_raw.wav", td / "vo.wav", cfg)
    check("voiceover normalised to ~-16 LUFS", abs(nv["after"]["I"] - (-16)) < 1.0 and nv["after"]["sample_peak"] <= -1.5 + 0.05,
          f"I={nv['after']['I']} LUFS, peak={nv['after']['sample_peak']} dBFS")
    res = mix_final_audio(td / "vo.wav", td / "music.wav", [(td / "sfx.wav", 1.0, None)], td / "mix.wav", cfg, duration=cfg.duration)
    mp = probe(td / "mix.wav")
    check("mix: 60 s, 48 kHz, stereo", abs(mp["duration"] - 60) < 0.05 and mp["sample_rate"] == 48000 and mp["channels"] == 2, str(mp))
    check("mix: true peak <= -1.0 dBTP", res["TP"] <= -1.0, f"TP={res['TP']} I={res['I']} LUFS")

print("\nRESULT:", "ALL PASS" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)

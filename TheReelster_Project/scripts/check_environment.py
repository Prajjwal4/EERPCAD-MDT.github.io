#!/usr/bin/env python3
"""Environment check for the TheReelster pipeline.
Run from anywhere: python3 scripts/check_environment.py
Writes qa/environment_report.txt. Exit code 0 only if every MANDATORY check passes."""
import importlib, os, platform, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "qa" / "environment_report.txt"
MIN_FREE_GB = 5.0
PY_PKGS = {"numpy": "numpy", "PIL": "Pillow", "moviepy": "moviepy", "pydub": "pydub"}
results = []  # (level, name, ok, detail)

def run(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:
        return 127, f"{type(e).__name__}: {e}"

def rec(level, name, ok, detail=""):
    results.append((level, name, ok, detail))

# 1. OS
rec("MANDATORY", "Operating system", True, f"{platform.system()} {platform.release()} ({platform.machine()}) | {platform.platform()}")

# 2/3. FFmpeg + ffprobe
for tool in ("ffmpeg", "ffprobe"):
    path = shutil.which(tool)
    if path:
        _, out = run([tool, "-version"])
        rec("MANDATORY", tool, True, f"{path} | {out.splitlines()[0]}")
    else:
        rec("MANDATORY", tool, False, "not found on PATH")

# libx264 / aac encoders and filters we rely on
if shutil.which("ffmpeg"):
    _, enc = run(["ffmpeg", "-hide_banner", "-encoders"])
    _, flt = run(["ffmpeg", "-hide_banner", "-filters"])
    for name, hay, needle in [("encoder libx264", enc, "libx264"), ("encoder aac", enc, " aac "),
                              ("filter drawtext", flt, "drawtext"), ("filter sidechaincompress", flt, "sidechaincompress"),
                              ("filter loudnorm", flt, "loudnorm"), ("filter zoompan", flt, "zoompan")]:
        rec("MANDATORY", f"ffmpeg {name}", needle in hay, "available" if needle in hay else "MISSING")

# 4/5. Python + pip
v = sys.version_info
rec("MANDATORY", "Python >= 3.9", v >= (3, 9), f"{sys.executable} | {sys.version.split()[0]}")
rc, out = run([sys.executable, "-m", "pip", "--version"])
rec("MANDATORY", "pip", rc == 0, out.splitlines()[0] if out else "not available")

# 6. Node / npm (informational: pipeline is Python + FFmpeg)
for tool in ("node", "npm"):
    path = shutil.which(tool)
    if path:
        _, out = run([tool, "--version"]); rec("OPTIONAL", tool, True, f"{path} | {out.splitlines()[0]}")
    else:
        rec("OPTIONAL", tool, False, "not found")

# 7. Disk space
free_gb = shutil.disk_usage(ROOT).free / 1e9
rec("MANDATORY", f"Free disk >= {MIN_FREE_GB:.0f} GB", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free at {ROOT}")

# Python packages (import-tested, not just pip-listed)
for mod, pipname in PY_PKGS.items():
    try:
        m = importlib.import_module(mod)
        from importlib.metadata import version as _v
        rec("MANDATORY", f"python package {pipname}", True, f"import ok, version {_v(pipname)}")
    except Exception as e:
        rec("MANDATORY", f"python package {pipname}", False, f"{type(e).__name__}: {e}")

# 8. Local image generation / rendering tools (only things that can be executed here)
try:
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    im = Image.new("RGB", (64, 64)); ImageDraw.Draw(im).text((0, 0), "x", font=f)
    rec("OPTIONAL", "Image rendering: Pillow + DejaVu Sans Bold", True, "can draw gradients/text/composites locally (procedural; NOT photoreal generation)")
except Exception as e:
    rec("OPTIONAL", "Image rendering: Pillow + DejaVu Sans Bold", False, f"{type(e).__name__}: {e}")
for tool in ("convert", "magick", "inkscape", "blender"):
    p = shutil.which(tool); rec("OPTIONAL", f"Image tool {tool}", bool(p), p or "not found")
pw = run(["node", "-e", "require('playwright');console.log('ok')"])[0] == 0 if shutil.which("node") else False
rec("OPTIONAL", "Playwright (HTML/CSS -> PNG rendering)", pw, "require('playwright') ok" if pw else "not importable")
rec("OPTIONAL", "AI image generation (local model)", False,
    "no local diffusion model/GPU tooling detected; script does not test remote tools. "
    "Remote generators (if any) must be verified by an actual generation call, not by this script.")

# FFmpeg 1080x1920 encode test (+ verify with ffprobe)
if shutil.which("ffmpeg") and shutil.which("ffprobe"):
    with tempfile.TemporaryDirectory() as td:
        out_mp4 = Path(td) / "test.mp4"
        rc, log = run(["ffmpeg", "-hide_banner", "-y", "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=2",
                       "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-shortest", str(out_mp4)])
        if rc == 0 and out_mp4.exists():
            _, pr = run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                         "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames", "-of", "csv=p=0", str(out_mp4)])
            ok = pr.startswith("h264,1080,1920,yuv420p,30/1,60")
            rec("MANDATORY", "FFmpeg 1080x1920 H.264 encode test", ok, f"ffprobe: {pr} (expected h264,1080,1920,yuv420p,30/1,60)")
        else:
            rec("MANDATORY", "FFmpeg 1080x1920 H.264 encode test", False, log[-500:])
else:
    rec("MANDATORY", "FFmpeg 1080x1920 H.264 encode test", False, "skipped: ffmpeg/ffprobe missing")

# Report
mand_fail = [r for r in results if r[0] == "MANDATORY" and not r[2]]
lines = [f"TheReelster environment report - {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "=" * 72]
for level, name, ok, detail in results:
    lines.append(f"[{'PASS' if ok else ('FAIL' if level == 'MANDATORY' else 'N/A ')}] ({level}) {name}\n        {detail}")
lines += ["=" * 72, f"RESULT: {'ALL MANDATORY CHECKS PASSED' if not mand_fail else 'FAILED: ' + ', '.join(r[1] for r in mand_fail)}"]
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if mand_fail else 0)

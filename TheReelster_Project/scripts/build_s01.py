#!/usr/bin/env python3
"""Build scene S01 -> scenes/S01.mp4, qa/S01_preview.png, and verify it.

Requires assets/visuals/visual_01.png. If it is missing this script stops: it never
substitutes another image (project rule).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image

from caption_renderer import CaptionGroup
from config import load_config
from scene_renderer import Camera, CaptionEvent, FFmpegWriter, Scene, SceneRenderer

cfg = load_config()
IMAGE = cfg.path("assets/visuals/visual_01.png")
OUT = cfg.path("scenes/S01.mp4")
PREVIEW = cfg.path("qa/S01_preview.png")

NARRATION = "Your mind has weaknesses you probably don't even know about."
HIGHLIGHTS = ("weaknesses",)

# 10 words > 7-word limit, so the sentence is shown as two caption groups.
# Frame windows are PROVISIONAL (estimated speech pace, ~2.8 words/s from ~0.3 s);
# they must be re-timed from the real voiceover transcription.
GROUPS = [("Your mind has weaknesses", 9, 54),
          ("you probably don't even know about.", 54, 138)]


def build_scene() -> Scene:
    if not IMAGE.is_file():
        sys.exit(f"MISSING ASSET: {IMAGE}\nNot substituting anything. Generate/provide visual_01.png first.")
    assert " ".join(g for g, _, _ in GROUPS) == NARRATION, "caption text must equal the narration"
    img = Image.open(IMAGE)
    ratio = img.width / img.height
    if abs(ratio - 9 / 16) > 0.02:
        print(f"WARNING: visual_01.png is {img.size} (aspect {ratio:.3f}, expected 0.5625). "
              "It will be cover-cropped to 9:16.", file=sys.stderr)
    return Scene(
        image=IMAGE,
        camera=Camera(zoom_start=1.00, zoom_end=1.08, ease="linear"),   # slow 100% -> 108%
        captions=[CaptionEvent(CaptionGroup(t, HIGHLIGHTS), s, e, kind="caption", position="lower") for t, s, e in GROUPS],
    )


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return {**j["streams"][0], "format_duration": float(j["format"]["duration"])}


def main():
    scene = build_scene()
    R = SceneRenderer(cfg)
    with FFmpegWriter(OUT, cfg) as w:
        n = R.render_scene(scene, w)          # strict: raises unless exactly 150 frames
    p = ffprobe(OUT)
    print(json.dumps(p, indent=2))
    ok = (n == 150 and int(p["nb_read_frames"]) == 150 and (p["width"], p["height"]) == (1080, 1920)
          and p["r_frame_rate"] == "30/1" and abs(p["format_duration"] - 5.0) < 0.01)
    # Preview: a frame where the highlighted caption is fully visible, extracted from the encoded video.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(OUT), "-vf", "select=eq(n\\,40)",
                    "-frames:v", "1", str(PREVIEW)], check=True)
    print("preview:", PREVIEW, Image.open(PREVIEW).size)
    print("S01 VERIFIED" if ok else "S01 FAILED VERIFICATION")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final assembly gate for TheReelster.

Assembly runs ONLY when every scene is individually approved (project_config.json -> script.scenes[*].approved) and
every scenes/SXX.mp4 exists with the exact format (150 frames, 1080x1920, 30 fps). Until then this script only reports
status and exits non-zero. The assembly itself (concat, captions carry-over, audio mix with ducked music + SFX, full-length
preview, QA report) is implemented once the scenes exist; the gate below is what prevents early or partial output.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qa_scene
from config import load_config


def gate(cfg) -> list[str]:
    problems = []
    for s in cfg.raw["script"]["scenes"]:
        sid = s["id"]
        video = cfg.path(f"scenes/{sid}.mp4")
        if not s.get("approved"):
            problems.append(f"{sid}: not approved by the user yet")
            continue
        if not video.is_file():
            problems.append(f"{sid}: approved but {video.name} is missing")
        elif not qa_scene.format_ok(qa_scene.probe(video)):
            problems.append(f"{sid}: {video.name} is not 150 frames / 1080x1920 / 30 fps")
    return problems


def main() -> int:
    cfg = load_config()
    problems = gate(cfg)
    n_ok = len(cfg.raw["script"]["scenes"]) - len(problems)
    print(f"scenes ready: {n_ok}/{len(cfg.raw['script']['scenes'])}")
    if problems:
        print("NOT assembling the final video yet:")
        for p in problems:
            print("  -", p)
        return 1
    print("All 12 scenes approved and present. Final assembly would start now.")
    raise SystemExit("assembly step not implemented yet")   # replaced when the scenes exist


if __name__ == "__main__":
    sys.exit(main())

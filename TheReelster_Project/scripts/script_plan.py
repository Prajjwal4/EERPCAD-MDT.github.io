"""The approved SCRIPT is the reference for scene timing.

Reads  project_config.json -> "script"   (scene windows, narration, delivery, caption groups, highlights)
       assets/audio/voiceover_timing.json (real word times, from build_voiceover.py)
and answers, for any scene n:
    * its window  [(n-1)*5, n*5] s  =  150 frames, global frame offset (n-1)*150
    * the words of its own line in SCENE-LOCAL frames (to anchor animation beats)
    * the caption events to draw in that scene, including lines that CROSS a scene boundary:
        - carry_in : a group already on screen when the scene starts (end of the previous line)
        - carry_out: a group still on screen when the scene ends (it continues in the next scene)
Nothing here guesses: if the script and the audio disagree, load fails with a clear message.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from config import Config, load_config

HOLD = 14            # frames a finished caption group stays up after its last word
HANDOFF = 1          # a group that follows another starts 1 frame after its first word (clean cut between groups)
CARRIED = -100       # word_start for words already revealed in the previous scene


def _pieces(word: str) -> int:
    """How many aligner tokens one script word spans (hyphenated words are split by the aligner)."""
    t = re.sub(r"[^\w' -]", "", word.lower()).replace("-", " ")
    return len([x for x in (w.strip("'") for w in t.split()) if x])


def _norm_text(t: str) -> str:
    return t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


@dataclass
class CaptionPlan:
    text: str
    highlights: tuple[str, ...]
    start: int                  # scene-local first frame (0 if carried in)
    end: int                    # scene-local end frame (<=150)
    word_starts: list[int]      # scene-local; CARRIED for words revealed in the previous scene
    carry_in: bool
    carry_out: bool
    line_scene: int             # which scene's line this group belongs to


@dataclass
class ScenePlan:
    number: int
    id: str
    title: str
    window: tuple[float, float]
    frame0: int
    frames: int
    narration: str | None
    delivery: str
    highlight: list[str]
    on_screen_text: list[str]
    words: list[dict] = field(default_factory=list)       # own line, scene-local frames (may exceed `frames` when it spills)
    line_start: float | None = None
    line_end: float | None = None
    captions: list[CaptionPlan] = field(default_factory=list)
    approved: bool = False

    @property
    def spill_out(self) -> float:
        return max(0.0, (self.line_end or 0) - self.window[1])

    @property
    def lead(self) -> float:
        return (self.line_start - self.window[0]) if self.line_start is not None else 0.0


class PlanError(ValueError):
    pass


class Plan:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        raw = self.cfg.raw["script"]
        self.title = raw["title"]
        self.raw_scenes = raw["scenes"]
        fps, sf = self.cfg.fps, self.cfg.scene_frames
        if len(self.raw_scenes) != self.cfg.scene_count:
            raise PlanError("script scene count != config scene count")
        for s in self.raw_scenes:
            n = s["number"]
            if s["window"] != [(n - 1) * 5, n * 5]:
                raise PlanError(f"{s['id']}: window {s['window']} is not the script's 5 s grid")
        self.timing = json.load(open(self.cfg.path("assets/audio/voiceover_timing.json")))
        if self.timing["fps"] != fps:
            raise PlanError("timing fps != project fps")
        self.sf = sf
        self._lines = {l["scene"]: l for l in self.timing["lines"]}
        self._merge_words()
        self._check_audio_matches_script()
        self._groups = self._build_groups()

    # ------------------------------------------------------------------ checks
    def _merge_words(self):
        """Re-express each line's timed tokens on the script's own words (one timing per script word)."""
        for s in self.raw_scenes:
            line = self._lines.get(s["number"])
            if not line or not s["narration"]:
                continue
            shown, toks, i, merged = s["narration"].split(), line["words"], 0, []
            if sum(_pieces(w) for w in shown) != len(toks):
                raise PlanError(f"{s['id']}: script has {sum(_pieces(w) for w in shown)} spoken tokens, "
                                f"voiceover timing has {len(toks)}")
            for w in shown:
                k = _pieces(w)
                a, b = toks[i], toks[i + k - 1]
                merged.append({"word": w, "start_frame": a["start_frame"], "end_frame": b["end_frame"],
                               "start": a["start"], "end": b["end"]})
                i += k
            line["words"] = merged

    def _check_audio_matches_script(self):
        for s in self.raw_scenes:
            n, narr = s["number"], s["narration"]
            line = self._lines.get(n)
            if narr is None:
                if line is not None:
                    raise PlanError(f"{s['id']} has no narration in the script but the voiceover has a line for it")
                continue
            if line is None:
                raise PlanError(f"{s['id']}: voiceover timing has no line")
            if _norm_text(line["text"]) != _norm_text(narr):
                raise PlanError(f"{s['id']}: voiceover text differs from the approved script:\n  {line['text']}\n  {narr}")
            if len(line["words"]) != len(narr.split()):
                raise PlanError(f"{s['id']}: {len(line['words'])} timed words vs {len(narr.split())} script words")
            if " ".join(s["caption_groups"]) != narr:
                raise PlanError(f"{s['id']}: caption_groups do not spell the narration exactly")
            if any(len(g.split()) > 7 for g in s["caption_groups"]):
                raise PlanError(f"{s['id']}: caption group over 7 words")
            w0 = s["window"][0]
            if not (w0 - 0.75 <= line["start"] <= w0 + 1.5):
                raise PlanError(f"{s['id']}: line starts at {line['start']}s, too far from its window start {w0}s")
        last = max(l["end"] for l in self.timing["lines"])
        if last > 55.0:
            raise PlanError(f"narration ends at {last}s - scene 12 must be voice-free (from 55 s)")

    # ------------------------------------------------------------------ groups
    def _build_groups(self) -> list[dict]:
        """Global caption groups for every narrated line, in time order, with global frame windows."""
        gl = []
        for s in self.raw_scenes:
            if s["narration"] is None:
                continue
            words = self._lines[s["number"]]["words"]
            i = 0
            for g in s["caption_groups"]:
                n = len(g.split())
                ws = words[i:i + n]
                i += n
                gl.append({"text": g, "scene": s["number"], "highlights": tuple(s["highlight"]),
                           "words": [(w["start_frame"], w["end_frame"]) for w in ws]})
        gl.sort(key=lambda g: g["words"][0][0])
        for k, g in enumerate(gl):
            g["start"] = g["words"][0][0] + (HANDOFF if k > 0 else 0)
            g["word_starts"] = [g["start"]] + [a for a, _ in g["words"][1:]]
        for k, g in enumerate(gl):
            end = g["words"][-1][1] + HOLD
            if k + 1 < len(gl):
                end = min(end, gl[k + 1]["start"])
            g["end"] = end
        return gl

    # ------------------------------------------------------------------ scenes
    def scene(self, n: int) -> ScenePlan:
        raw = self.raw_scenes[n - 1]
        f0 = (n - 1) * self.sf
        sp = ScenePlan(n, raw["id"], raw["title"], tuple(raw["window"]), f0, self.sf, raw["narration"], raw["delivery"],
                       list(raw["highlight"]), list(raw.get("on_screen_text", [])), approved=raw.get("approved", False))
        line = self._lines.get(n)
        if line:
            sp.line_start, sp.line_end = line["start"], line["end"]
            sp.words = [{"word": w["word"], "start": w["start_frame"] - f0, "end": w["end_frame"] - f0} for w in line["words"]]
        for g in self._groups:
            if g["start"] < f0 + self.sf and g["end"] > f0:
                carry_in, carry_out = g["start"] < f0, g["end"] > f0 + self.sf
                ws = [(w - f0) if w >= f0 - 5 else CARRIED for w in g["word_starts"]]      # a word revealed <=5 frames before the cut is still fading in
                sp.captions.append(CaptionPlan(g["text"], g["highlights"], max(g["start"] - f0, 0), min(g["end"] - f0, self.sf),
                                               ws, carry_in, carry_out, g["scene"]))
        return sp

    # ------------------------------------------------------------------ report
    def report(self) -> str:
        out = [f"{'scene':<6}{'window':>10} {'line (s)':>14} {'lead':>6} {'spill':>6}  caption groups (scene-local frames)"]
        for s in self.raw_scenes:
            sp = self.scene(s["number"])
            line = f"{sp.line_start:5.2f}-{sp.line_end:5.2f}" if sp.line_start is not None else "   (no voice)"
            caps = "; ".join(f"{'<' if c.carry_in else ''}{c.start}-{c.end}{'>' if c.carry_out else ''}" for c in sp.captions)
            out.append(f"{sp.id:<6}{sp.window[0]:>4.0f}-{sp.window[1]:<4.0f}s {line:>14} {sp.lead:+6.2f} {sp.spill_out:6.2f}  {caps}")
        return "\n".join(out)


def load_plan() -> Plan:
    return Plan()


def load_scene(n: int) -> ScenePlan:
    return Plan().scene(n)


if __name__ == "__main__":
    print(Plan().report())

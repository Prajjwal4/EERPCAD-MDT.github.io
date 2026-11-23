"""Audio helpers (FFmpeg is the mixing engine): probe, loudness measurement, voiceover
normalisation, and the final music-ducked mix. Targets come from project_config.json.

Interpretation notes (the spec is terse):
  * voiceover_target_lufs  -> integrated loudness of the narration after normalisation.
  * music_target_db        -> integrated loudness (LUFS) of the music bed *before* ducking.
  * true_peak_max_dbtp     -> verified on the final mix; the limiter ceiling is set lower
                              (audio_mix.limiter_ceiling_dbfs) to leave inter-sample margin.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from config import Config, load_config


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}... failed:\n{p.stderr[-1500:]}")
    return p.stderr + p.stdout


def probe(path: Path | str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=sample_rate,channels:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return {"duration": float(j["format"]["duration"]),
            "sample_rate": int(j["streams"][0]["sample_rate"]), "channels": int(j["streams"][0]["channels"])}


def measure_loudness(path: Path | str, filters: str = "") -> dict:
    """Integrated loudness (LUFS), true peak (dBTP), LRA via loudnorm analysis, plus sample peak (dBFS)."""
    pre = filters + "," if filters else ""
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                f"{pre}loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json,volumedetect", "-f", "null", "-"])
    j = json.loads(re.findall(r"\{[^{}]*\}", log, re.S)[-1])
    mv = re.search(r"max_volume:\s*(-?[\d.]+) dB", log)
    return {"I": float(j["input_i"]), "TP": float(j["input_tp"]), "LRA": float(j["input_lra"]),
            "thresh": float(j["input_thresh"]), "offset": float(j["target_offset"]),
            "sample_peak": float(mv.group(1)) if mv else float("nan")}


def normalize_voiceover(src: Path | str, dst: Path | str, cfg: Config | None = None) -> dict:
    """Two-pass loudnorm to the configured LUFS with true peak <= voiceover_peak_dbfs_max."""
    cfg = cfg or load_config()
    a = cfg.audio
    target, tp = a["voiceover_target_lufs"], a["voiceover_peak_dbfs_max"]
    m = measure_loudness(src)
    af = (f"loudnorm=I={target}:TP={tp}:LRA=11:measured_I={m['I']}:measured_TP={m['TP']}:measured_LRA={m['LRA']}"
          f":measured_thresh={m['thresh']}:offset={m['offset']}:linear=true,"
          f"aresample={a['sample_rate']},aformat=channel_layouts=stereo")
    _run(["ffmpeg", "-hide_banner", "-y", "-i", str(src), "-af", af, "-c:a", "pcm_s16le", "-ar", str(a["sample_rate"]), str(dst)])
    after = measure_loudness(dst)
    if after["sample_peak"] > tp + 0.05:
        raise RuntimeError(f"voiceover peak {after['sample_peak']} dBFS exceeds {tp}")
    return {"before": m, "after": after}


def mix_final_audio(voiceover: Path | str, music: Path | str, sfx: list[tuple[Path | str, float, float | None]],
                    out: Path | str, cfg: Config | None = None, duration: float | None = None) -> dict:
    """voiceover + ducked music + timed SFX -> `out` (wav).

    sfx: list of (file, start_seconds, gain_db or None for config default).
    Music: normalised to music_target_db LUFS, faded in/out, then sidechain-ducked by the voiceover.
    """
    cfg = cfg or load_config()
    a, mx = cfg.audio, cfg.audio_mix
    D, sr = duration or cfg.duration, a["sample_rate"]
    fmt = f"aresample={sr},aformat=sample_fmts=fltp:channel_layouts=stereo"

    music_gain = a["music_target_db"] - measure_loudness(music)["I"]
    inputs = ["-i", str(voiceover), "-i", str(music)]
    for f, _, _ in sfx:
        inputs += ["-i", str(f)]

    g = [f"[0:a]{fmt},apad=whole_dur={D},atrim=0:{D},asplit=2[vo][vokey]"]
    fi, fo = a["music_fade_in_seconds"], a["music_fade_out_seconds"]
    g.append(f"[1:a]{fmt},volume={music_gain:.3f}dB,aloop=loop=-1:size=2147483647,atrim=0:{D},"
             f"afade=t=in:st=0:d={fi},afade=t=out:st={D - fo}:d={fo}[mus]")
    if a["music_ducking"]:
        g.append(f"[mus][vokey]sidechaincompress=threshold={mx['duck_threshold']}:ratio={mx['duck_ratio']}"
                 f":attack={mx['duck_attack_ms']}:release={mx['duck_release_ms']}:makeup=1[duck]")
        music_label = "duck"
    else:
        music_label = "mus"
    labels = ["vo", music_label]
    for k, (_, start, gain) in enumerate(sfx):
        ms = int(round(start * 1000))
        g.append(f"[{k + 2}:a]{fmt},volume={gain if gain is not None else mx['sfx_gain_db']}dB,adelay={ms}|{ms}[sfx{k}]")
        labels.append(f"sfx{k}")
    ceiling = 10 ** (mx["limiter_ceiling_dbfs"] / 20)
    g.append("".join(f"[{l}]" for l in labels) +
             f"amix=inputs={len(labels)}:normalize=0:duration=longest,atrim=0:{D},alimiter=limit={ceiling:.4f}:level=disabled[out]")

    _run(["ffmpeg", "-hide_banner", "-y", *inputs, "-filter_complex", ";".join(g), "-map", "[out]",
          "-c:a", "pcm_s16le", "-ar", str(sr), "-ac", "2", str(out)])
    res = measure_loudness(out)
    if res["TP"] > a["true_peak_max_dbtp"]:
        raise RuntimeError(f"final mix true peak {res['TP']} dBTP exceeds {a['true_peak_max_dbtp']}")
    return res

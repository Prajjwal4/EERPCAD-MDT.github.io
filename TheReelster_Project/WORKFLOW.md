# TheReelster production workflow (local code only)

Format: 1080x1920 (9:16), 30 fps, YouTube Shorts. 12 scenes x 150 frames = 1800 frames.

## Rules
- No Higgsfield. No external image/video generation of any kind.
- ElevenLabs: voiceover only (text-to-speech via the connector, one take per line). Never images, video or transcription. Takes live in `assets/audio/takes/`; `scripts/build_voiceover.py` times and assembles them into `assets/audio/voiceover_01.wav` + `voiceover_timing.json`.
- Everything visual is rendered locally: Python + NumPy + Pillow draw each frame, FFmpeg encodes.

## Pipeline
1. `scripts/check_environment.py` - dependencies + FFmpeg encode test.
2. Per scene `scripts/scene_SXX.py` - procedural plate (`assets/visuals/visual_XX.png`, native 1080x1920) + per-frame animation -> `scenes/SXX.mp4` (150 frames) + `qa/SXX_preview.png`.
3. Captions/kinetic type and scene beats are timed to `voiceover_timing.json` (word-level, local forced alignment).
4. Assemble 12 scenes, mix audio (voiceover -16 LUFS, ducked music, SFX) with FFmpeg -> `output/`.
5. QA report: frame count, dimensions, loudness, safe margins.

## Missing-asset policy
A missing asset stops the build. Nothing is silently substituted.

## Final assembly (gated)
When - and only when - all 12 scenes are individually completed, reviewed and approved by the user, the project is assembled
automatically into ONE video: `output/TheReelster_Final.mp4` (1080x1920, 30 fps, H.264, full ElevenLabs narration, consistent captions,
transitions, sound effects and music) plus a full-length preview and an automated QA report. `scripts/assemble_final.py` enforces the gate:
it refuses to run unless every scene is approved (`script.scenes[*].approved`) and every `scenes/SXX.mp4` exists and passes its checks.
The project is not declared complete until the automated quality checks in `final_assembly` all pass.

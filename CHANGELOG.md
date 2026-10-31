# Changelog

All notable changes to `/watch` are documented here.

## [0.8.0] — 2026-10-31

### Added
- **Shape-change measurements.** `track --distance a,b` follows the distance between two tracked points over time (a mouth opening, a gap), `--angle a,b,c` the angle at the middle of three (a fin, a limb, a hinge). The report gives min/max/mean, the maxima and minima with their times, the number of cycles, the period and frequency, a curve, and the values over time; `plot.png` and `summary.png` get a curve per measurement, and `tracked.mp4` draws the line between the points with the live value. Checked on a synthetic mouth (period 1.00 s for a true 1.0 s, maxima within 0.03 s) and hinge (19–82° for a true 20–80°, period 2.03 s for 2.0 s).
- `SKILL.md` "Which tool for which question": looking at frames is not an analysis; measure with `track` and use `sheet` only to find the moment and the coordinates. `sheet` images and reports now say "look only".

### Fixed
- A failure while rendering the human outputs (summary picture, annotated video) no longer aborts the whole track report.

## [0.7.0] — 2026-10-31

### Added
- **CoTracker3 point tracking.** `track --obj NAME@T=x,y` now follows exactly the clicked pixel with Meta's CoTracker3 (online) point-tracking model, forward and backward in time, with per-frame visibility, camera-motion removal (scene coordinates) and loss at scene cuts. `--engine auto|point|box` selects it; the box engine (ViT) stays for `x,y,w,h` seeds, flat uniformly coloured objects and size/squash-and-stretch analysis. Checked on a synthetic bouncing ball: bounce times within one frame, bounce ratio 72–74 % for a true 72 %, gravity 2.49 for a true 2.5, identical with a panning camera.
- `setup.py --install-tracker` creates a managed PyTorch venv, loads CoTracker3 at a pinned commit and verifies the weights' SHA-256. `setup.py --json` reports `tracker_ready`. Without it `auto` falls back to the box engine and says so.
- CoTracker3 is licensed CC BY-NC 4.0 (non-commercial); the report names it.

## [0.6.0] — 2026-10-31

### Changed
- **The agent chooses the object, the tool detects nothing.** `pick` no longer proposes numbered candidates; it renders a frame with a coordinate grid. `pick --at x,y` draws a cross and a magnified patch so a click can be verified, `--box` zooms. `track --obj NAME@T=x,y --size S` follows exactly the clicked spot (a box of width S frame widths centred on it); `x,y,w,h` gives an explicit box. Region growing, default boxes and `#N` candidates were removed.

## [0.5.1] — 2026-10-31

### Added
- **Outputs made for humans.** `--motion` writes `motion_view.mp4`: the video with cuts (red frame), a camera arrow, moving regions, tracked movers (A, B, …) with trails and a scrolling activity gauge (cuts, audio beats). `track.py track` writes `summary.png` (key facts, path, filmstrips, curves in one picture) and a richer `tracked.mp4` (BOUNCE 72 % / TURN / PAUSE flashes, speed gauge with cursor). They are marked FOR THE HUMAN in the file list. `--no-view` skips the motion video.
- `SKILL.md` section "Show your work to the user": say which tool is running, deliver the human outputs, then answer with the measured numbers.
- `sheet` images carry a "WATCH SHEET" banner so they are not mistaken for plain frame sampling.

### Fixed
- Text in rendered images was ghosted (an outline of different advance under the text) and broke on "°"; labels are now ASCII-safe with a same-stroke outline.

## [0.5.0] — 2026-10-31

### Added
- **Object tools (`track.py`).** `pick` renders a frame with a labelled 0..1 coordinate grid and numbered object candidates (moving regions and distinct colour regions); `track` follows one or several chosen objects forward and backward with a small neural ViT tracker (OpenCV `TrackerVit`, 0.7 MB model, checksum-pinned) plus camera-motion estimation, automatic working resolution for tiny objects, confidence per frame and re-finding by appearance after occlusion or a scene cut.
- Path analysis as text and images: position/size/speed table, ASCII path map, CSV, pauses, floor/ceiling/wall bounces with the speed kept (from fitted parabolas), turns, free-flight gravity, periodic motion (frequency, amplitude), easing of the main move, size and squash-and-stretch, and relations between objects (distance, closest approach, contact). Camera motion is removed so paths are in scene coordinates when the camera moves.
- Images: `paths.png` (time-coloured trail, dots at fixed time steps, event rings, canvas that grows beyond the frame), `filmstrip_*.png` (object crops over time), `plot.png`, `seed_*.png`; `tracked.mp4` with box, trail and speed drawn on the video for the human viewer.
- `sheet` (slow-motion contact sheet, optional zoom box and frame-coordinate grid) and `flow` (motion arrows and flow colours with a per-region heading and speed list).
- The `--motion` report prints a ready-to-run `track.py --obj` line for every tracked mover. `setup.py --install-motion` also fetches the tracking model; `setup.py --json` reports `track_model_ready`.

## [0.4.0] — 2026-10-31

### Added
- **Motion mode (`--motion`).** Measures how a video moves instead of sampling stills. One local pass computes cuts and shot lengths, camera behaviour per shot (pan, tilt, zoom, roll, shake) from RANSAC on tracked features, camera-compensated dense optical flow, a local-motion measure that catches small fast objects, and audio onsets, tempo and loudness. Moving objects are tracked with feature points and with a moving-region tracker, clustered, and described with start/end, direction, peak speed, a speed profile and a fitted easing curve (linear, ease-in/out, smoothstep, overshoot/spring). A shuffle test says whether cuts and motion peaks land on the beat more often than chance.
- Images that replace dozens of stills: `overview.png` (thumbnails plus motion, camera and audio curves on one time axis), per-shot camera-stabilised motion history (colour = time), strobe (chronophotography) and track paths with 0.1 s dots. Panning shots use their best low-drift window; overlays are skipped, and the report says so, when the camera moves too much.
- `setup.py --install-motion` creates a managed `~/.cache/watch/motion-venv` (numpy, OpenCV, SciPy; no models). `setup.py --json` reports `motion_ready`. `WATCH_MOTION_VENV` relocates it. The base skill stays standard-library only.
- `--max-shots N` sets how many shots get detail images.

## [0.3.2] — 2026-09-25

### Changed
- Setup lets the user choose how to add an API key: paste it in chat, or have the agent open `~/.config/watch/.env` in their text editor so they can paste it themselves.
- README: Watch is supported in Claude Code (Claude Desktop's Code tab, VS Code, or a terminal). Claude Chat and Cowork are not supported: in Cowork's cloud environment the Gemini key does not persist between tasks and most sites block yt-dlp downloads. The standalone Chat upload instructions are removed.

## [0.3.1] — 2026-09-25

### Fixed
- `SKILL.md` frontmatter uses only Agent Skills keys, so `watch.skill` uploads to Claude (the upload rejected `version`, `argument-hint`, `user-invocable`, and other extra keys). The version moves to `metadata.version`.

## [0.3.0] — 2026-09-25

### Added
- **Gemini engine.** With a `GEMINI_API_KEY`, watch sends the video to Google's agentic video-understanding model (`gemini-3.7-flash`) and relays its timestamped answer: YouTube URLs directly, everything else via a streamed Files API upload that is deleted afterwards. New `--engine auto|gemini|local`, `--question`, `WATCH_ENGINE`, `WATCH_GEMINI_MODEL`, `WATCH_GEMINI_TIMEOUT`. `--start/--end` use static clipping. Standard library only; no silent fallback between engines.
- Setup wizard asks for the engine first; a Gemini-only setup no longer requires `ffmpeg`/`yt-dlp` for YouTube URLs.
- Managed, optional WhisperX 3.8.6 setup with uv, a separate Python 3.12 environment, cache warm-up, recoverable installation state, and local CPU transcription. Default model: small, int8, batch 8; language and timeout are configurable. Local failures never select a cloud provider.
- One-time fallback selection in the skill wizard: WhisperX, Groq, OpenAI, or captions only. Existing installations retain the `auto` cloud-key preference.
- Original-language caption selection and provenance (English preferred when the source language is unknown), `--sub-lang`, and explicit cookie-file/browser options shared across downloader stages.
- Reports for unavailable modalities, no speech, and missing cloud-transcription intervals. Offline dependency diagnostics and a cross-platform CI gate for release publication.

### Fixed
- SessionStart hook command now quotes `${CLAUDE_PLUGIN_ROOT}`, so plugin paths containing spaces work.
- Uniform sampling now spans the actual capped range and labels selected source frames with their real timestamps, including fractional seeks and cue frames.
- FFmpeg sync-option capability detection (including FFmpeg 9's `-fps_mode[:<stream_spec>]` help format; tested on FFmpeg 4.4 through 9.0), empty keyframe-window fallback, RGB thumbnail deduplication, and actionable probe/launch errors.
- UTF-8 output and subprocess decoding, BOM-aware shared config parsing, provider-key pairing, atomic config updates, keyless readiness, and advisory setup hooks.
- Bounded caption downloads respect watch-owned output templates while retaining intentional yt-dlp configuration. Fresh run directories and completed final-path validation prevent stale/partial media reuse.
- WebVTT optional-hour timestamps, entities, cue/block parsing, and overlap-scoped deduplication preserve independent repeated speech.
- Full-track caption availability is checked before focus filtering; successful captions survive media/probe failures.
- Conservative 24,000,000-byte cloud upload budgets, actual chunk/multipart size checks, and explicit partial/no-speech metadata.

### Changed
- Only the latest yt-dlp release is supported. An HTTP 403 now tells the agent to update yt-dlp with its owning package manager and retry once; CI tests against the latest release instead of a pinned one.
- First-success installation documentation separates media tools from optional speech backends and describes host/network limitations.
- User-provided output directories now contain a disposable run child; cleanup preserves the parent and source files.
- The base runtime remains standard-library-only; local model environments and caches live outside the self-contained skill folder.

## [0.2.0] — 2026-06-29

### Added
- **`--detail` dial** with four modes — `transcript` (captions only, no frames), `efficient` (fast keyframe pass, cap 50), `balanced` (scene-aware, cap 100, default), and `token-burner` (scene-aware, uncapped). Set the default with `WATCH_DETAIL` in `~/.config/watch/.env`.
- **Frame deduplication** (default on; `--no-dedup` to disable). Before the budget cap, a pass downscales each frame to a 16×16 grayscale thumbnail and drops frames whose mean per-pixel difference from the last *kept* frame is within threshold — so the budget goes to distinct content instead of held slides and static recordings. The **Frames** report line shows how many near-duplicates were dropped.
- **Whisper auto-chunking.** Audio over the 25 MB upload cap is split into evenly sized chunks, transcribed per chunk, with segment timestamps shifted back into source time. Partial failures are tolerated — transcription only fails if *every* chunk fails, so length alone no longer breaks it.
- **`--timestamps T1,T2,…`** — grab a frame at each absolute timestamp; reserved against the cap, and the only frames produced under `--detail transcript`.
- **`--no-whisper`** — disable speech fallback while retaining native captions.
- pytest suite covering config, dedup, download, fixtures, frames, setup, timestamps, watch, and whisper (no network; ffmpeg-synthesized clips).

### Changed
- **Restructured into a self-contained `skills/watch/` package** so `SKILL.md` and its `scripts/` runtime are siblings in one folder. This fixes installs on Codex, Cursor, Copilot, and other Agent Skills hosts: `npx skills add` now copies the skill as a working unit instead of grabbing the root `SKILL.md` without its scripts.
- **Harness-agnostic path resolution** — `SKILL.md` resolves `$SKILL_DIR` from where it was Read instead of the Claude-Code-only `${CLAUDE_SKILL_DIR}`, so script calls work on every host.
- `/watch` is now derived from `SKILL.md` frontmatter; the separate `commands/watch.md` wrapper was dropped to avoid a duplicate slash command.
- `balanced` now full-decodes to detect every scene cut across the whole video. The previous early-exit was faster but kept only the first cuts and dropped the tail of long videos.
- `token-burner` is exempt from the long-video "sparse scan" warning, since it keeps every scene-change frame.
- `--max-frames` is now an override on top of each mode's default cap, rather than a fixed default of 80.

### Fixed
- Non-Claude installs (`npx skills add`) were dead on arrival — the installer copied `SKILL.md` without the `scripts/` it shells out to. The self-contained package layout resolves this.

### Removed
- `V2_PLAN.md` and `V2_CONCERNS.md` planning docs.

## [0.1.3] — 2026-05-09

### Fixed
- Windows: `video.info.json` is read as UTF-8 (#4). Previously `Path.read_text()` defaulted to cp1252 on Windows and crashed on yt-dlp's UTF-8 output, silently dropping Title/Uploader from the report. Same fix applied to `.env` reads/writes in `whisper.py` and `setup.py`.
- `download.py` now logs info.json parse failures to stderr instead of swallowing them.

### Security
- Hardened subprocess argv against option injection (#2): inserted `--` before the URL in the yt-dlp argv, and tightened `is_url` to reject `-`-prefixed sources and require a non-empty netloc. Resolved video/audio paths to absolute via `Path.resolve()` before passing to `ffmpeg`/`ffprobe`, so a relative path starting with `-` can't be misinterpreted as a flag.

## [0.1.2] — 2026-04-24

### Fixed
- Windows console crash: removed the emoji from the long-video warning in `watch.py`; cp1252 consoles couldn't encode it.
- `setup.py` now prints `winget` / `pip` install commands on Windows instead of "unsupported platform" — matches what the README already promised.

### Changed
- `SKILL.md` notes that on Windows the scripts must be invoked with `python`, not `python3` (the latter is the Microsoft Store stub on Windows).

## [0.1.1] — 2026-04-24

### Fixed
- Added `commands/watch.md` shim so `/watch` is callable when installed as a Claude Code plugin. Without it, the plugin loaded but the skill wasn't exposed as a slash command.
- `scripts/build-skill.sh` now strips `commands/` from the claude.ai `.skill` bundle alongside `hooks/` and `.claude-plugin/`.

## [0.1.0] — 2026-04-24

Initial marketplace release.

### Added
- `/watch <url-or-path> [question]` slash command.
- yt-dlp download with native caption extraction (manual + auto-subs).
- ffmpeg frame extraction with auto-scaled fps (≤2 fps, ≤100 frames, duration-aware budget).
- `--start` / `--end` focused mode with denser frame budget and transcript range filtering.
- Whisper fallback (Groq preferred, OpenAI secondary) for videos without captions.
- `setup.py` preflight: silent `--check`, structured `--json`, and installer that auto-runs `brew install` on macOS.
- Session-start hook that prints a one-line status on first run / partial config.
- `.skill` bundle packaging for claude.ai upload via `scripts/build-skill.sh`.

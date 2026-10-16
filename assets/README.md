# Assets Directory

This folder holds bundled commercial-free media and 3D character rigs, as well as any custom videos, still photographs, or `.fbx` characters you supply. All bundled runtime assets in this repository are **100% permissive and free for unrestricted commercial use** (see [`LICENCES.md`](LICENCES.md) and [`licences.json`](licences.json) for full provenance and attribution details).

Run `v2bm doctor` at any time to verify what is present and validate all rigs and media files.

## Directory Layout

| Path | Purpose | Details |
| :--- | :--- | :--- |
| `rigs/humanoid_rig.fbx` | Default Studio Humanoid Rig (`CC0-1.0`) | Procedurally generated via `scripts/generate_humanoid_rig.py` in Blender 5.2.2 LTS |
| `rigs/kenney_skater.fbx` | Kenney Skater Male Rig (`CC0-1.0`) | Clothed, UV-textured humanoid rig with companion `kenney_skater.png` diffuse atlas |
| `rigs/kenney_female.fbx` | Kenney Retro Female Rig (`CC0-1.0`) | Clothed, UV-textured humanoid rig with companion `kenney_female.png` diffuse atlas |
| `video/<your_clip>.mp4` | Input motion videos | Single clearly visible person (or multi-person when used with `--target-subject`) |
| `image/<your_photo>.png` | Input still photographs | Single clearly visible person for static 3D pose recovery (`--image`) |

`v2bm run` without flags uses the most recently added file in `assets/video/` and `assets/rigs/humanoid_rig.fbx`; pass `--video`, `--image`, or `--rig` to select inputs explicitly.

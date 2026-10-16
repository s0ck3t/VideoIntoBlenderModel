# AGENTS.md

VideoIntoBlenderModel drives an standard humanoid-rigged 3D character with human motion
extracted from a monocular video, or a static 3D pose from a still photo.
CLI: `v2bm`.

## Setup

```bash
pip install --upgrade pip setuptools wheel \
  && pip install -e .
```

All bundled rigs, weights, and example assets are 100% permissive and free for commercial use (see `assets/LICENCES.md`):

| What | Where to put it | Source |
| --- | --- | --- |
| Humanoid character | `assets/rigs/humanoid_rig.fbx` (or `kenney_skater.fbx`, `kenney_female.fbx`, `erika_archer.fbx`) | Bundled CC0 / Royalty-Free rigs, or any custom FBX Binary in T-pose |
| Input video | `assets/video/<name>.mp4` | Single clearly visible person (or multi-person with `--target-subject`) |
| Input still | `assets/image/<name>.png` | Single clearly visible person, head to toe (optional; `--image`) |

Inference weights (`weights/mediapipe/pose_landmarker_full.task`, ~9 MB, Apache-2.0) are verified or auto-downloaded on the first run.

## Commands

```bash
v2bm doctor        # verify assets + dependencies; prints what is missing and where to get it
v2bm run           # latest video in assets/video/ + assets/rigs/humanoid_rig.fbx
v2bm run --video assets/video/kung_fu_haikou.mp4 --rig assets/rigs/erika_archer.fbx
v2bm run --video assets/video/breakdancer_kickup.mp4 --rig assets/rigs/kenney_skater.fbx
v2bm run --image assets/image/urban_dancer_pose.png --rig assets/rigs/humanoid_rig.fbx
v2bm run --skeleton outputs/<run>/skeleton_motion.npz --rig ...   # reuse extraction, swap rig
v2bm run --video MULTI.mp4 --target-subject left --rig ...        # isolate performer in multi-person video
v2bm isolate --video MULTI.mp4 --output ISOLATED.mp4 --target-subject left
v2bm run --output-dir DIR --progress-jsonl FILE --no-preview      # Blender Extension / automation
```

`--video` / `--image` and `--rig` combine freely (not `--video` with `--image`).
Extraction is the slow step; reuse it with `--skeleton` when only the rig changes.

## Blender Extension

The add-on is a thin UI. It must **not** import MediaPipe / SciPy / OpenCV in
Blender’s Python. It launches the user’s venv:

```text
<external-python> -m v2bm.cli run ...
```

How to install and use it (for the user, or to walk them through it):
[`blender_extension/README.md`](blender_extension/README.md).

When helping a user:

1. They need the Python 3.10+ venv (`pip install -e .`) and a Humanoid FBX as in the CLI (see Setup above).
2. Build `dist/v2bm-0.1.0.zip` with
   `python scripts/package_blender_extension.py`.
3. In Blender 4.2+: **Edit → Preferences → Get Extensions**, then the **∨**
   next to Repositories → **Install from Disk…** (no large button). Enable it.
4. The panel is **3D Viewport → N sidebar → VideoIntoBlenderModel**, not Get
   Extensions. Setup is the first box: UI Language, External Python
   (`.venv/bin/python` or `Scripts\python.exe`, not Blender’s interpreter),
   Project Directory, Device, then **Check Environment**.
5. After add-on **code** changes they must rebuild the zip and reinstall.
   The installed copy is not live from this repo. If
   `already registered` / `EnumProperty could not register` appears: disable,
   **quit Blender fully**, reopen, install again.
6. Jobs write `<project>/outputs/blender_jobs/<job-id>/`. After import, hide
   **Icosphere** and the **armature** in the Outliner; leave **skinned_mesh**
   visible. FPS must match the source clip duration (a 60 fps video tagged as
   30 fps plays at half speed).

Smoke-test checklist: [`docs/blender-extension.md`](docs/blender-extension.md).

## Outputs

Each run writes `outputs/<YYYYMMDD_HHMMSS>_<source>/`:

- `run.json` — inputs, timings, output listing
- `skeleton_motion.npz` — 3D human skeleton (24 Body24 joints, Y-up metres)
- `rig_rotations.npz` — per-bone animation rotations (quaternion wxyz; convention string inside)
- `character_animation.glb` — skinned character + clip (glTF 2.0; Blender: File → Import → glTF 2.0)
- Video in: `videos/human_skeleton.mp4`, `videos/humanoid_skeleton.mp4`, `videos/humanoid_character.mp4`, `videos/compare.mp4` (2x2: original / humanoid skeleton / human skeleton / character)
- Photo in: the same four views as PNG under `images/`, plus `images/before_after_360_compare.mp4` (left original / right 10° orbit of the rig)

## Code map

- `v2bm/paths.py` — all filesystem locations
- `v2bm/extract.py` — video or still -> 3D human skeleton (MediaPipe Pose + IK optimiser)
- `v2bm/rig/` — FBX parsing, FK, retargeting, skinning, rendering
- `v2bm/pipeline.py` — one run end to end
- `v2bm/cli.py` — `v2bm doctor` / `v2bm run`
- `blender_extension/v2bm_blender/` — Blender 4.2+ Extension (UI + external process)
- `blender_extension/README.md` — install and use the add-on
- `docs/blender-extension.md` — maintainer smoke-test checklist
- `scripts/package_blender_extension.py` — build `dist/v2bm-0.1.0.zip`

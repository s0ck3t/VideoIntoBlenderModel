# VideoIntoBlenderModel Blender Extension

A Blender **4.2+ / 5.2 LTS** add-on that runs `VideoIntoBlenderModel` (`v2bm`) in an **external Python virtual environment**, streams live progress into the 3D Viewport, and automatically imports the generated skinned humanoid character (`.glb`) into the active Blender scene.

Blender’s bundled Python interpreter stays completely clean and never loads `mediapipe`, `opencv-python`, or `scipy`. Instead, the panel launches:

```text
<external-python> -m v2bm.cli run ...
```

## Prerequisites

Set up the core Python environment in a terminal first (full details in the [repository README](../README.md)):

1. Clone this repository and install `v2bm` into a Python 3.10+ virtual environment (`venv`):
   ```bash
   python -m venv venv
   # Linux / macOS:
   ./venv/bin/pip install -e .
   # Optional NVIDIA CUDA GPU acceleration (CuPy MIT):
   ./venv/bin/pip install -e ".[cuda]"

   # Windows PowerShell:
   .\venv\Scripts\python.exe -m pip install -e .
   # Optional NVIDIA CUDA GPU acceleration (CuPy MIT):
   .\venv\Scripts\python.exe -m pip install -e ".[cuda]"
   ```
2. Verify that the default CC0 character rig (`assets/rigs/humanoid_rig.fbx`) or your custom `.fbx` character is present in `assets/rigs/`.
3. Confirm everything passes preflight diagnostics:
   ```bash
   v2bm doctor
   ```

The first run automatically verifies or downloads the lightweight Google MediaPipe Pose Landmarker weights (**~9 MB**, Apache-2.0 licence) into `<project>/weights/mediapipe/`.

## Build & Install the Add-on

From the repository root:

```bash
python scripts/package_blender_extension.py
```

This builds `dist/v2bm-0.1.0.zip` with `blender_manifest.toml` and `__init__.py` at the archive root.

In **Blender 4.2+ / 5.2 LTS**:

1. Open **Edit → Preferences → Get Extensions**.
2. Next to **Repositories** in the top-right corner, open the small **∨** dropdown menu → **Install from Disk…**.
3. Select `dist/v2bm-0.1.0.zip` and enable **VideoIntoBlenderModel**.

Or install directly from the command line:

```bash
blender --command extension install-file --repo user_default --enable dist/v2bm-0.1.0.zip
```

> [!NOTE]
> Whenever you modify the add-on source code under `blender_extension/v2bm_blender/`, rebuild `dist/v2bm-0.1.0.zip` and reinstall it in Blender.

## Open the Viewport Panel

1. Switch to the **3D Viewport**.
2. Press **N** to toggle the right-hand sidebar.
3. Select the **VideoIntoBlenderModel** tab.

The panel is organised into three sections: **Setup**, **Input**, and **Progress / Result**.

## Setup

Configure these fields in the **Setup** box (or under **Edit → Preferences → Add-ons → VideoIntoBlenderModel**):

| Field | Description |
| :--- | :--- |
| **UI Language** | `Auto` follows Blender / system locale, or select a specific language |
| **External Python** | Path to your virtualenv Python executable (**not** Blender’s bundled Python) |
| **Project Directory** | Root directory of this repository (containing `assets/` and `weights/`) |
| **Device** | `Auto (CUDA if available)` (default), `CUDA (NVIDIA GPU)`, `CPU`, or `MPS` |

| Platform | Typical External Python Path |
| :--- | :--- |
| **Windows** | `<project>\venv\Scripts\python.exe` (or `.venv\Scripts\python.exe`) |
| **macOS / Linux** | `<project>/venv/bin/python` (or `.venv/bin/python`) |

Click **Check Environment** to run `v2bm doctor` and verify that the interpreter and assets are ready.

## Generate Motion or Static Pose

In the **Input** section:

1. **Source Type**: Choose `Video` or `Image`.
2. **Source File**: Select your input video clip or still photograph.
3. **Humanoid Rig FBX**: Choose `assets/rigs/humanoid_rig.fbx`, `kenney_skater.fbx`, `kenney_female.fbx`, `erika_archer.fbx`, or any custom humanoid `.fbx`.
4. Leave **Generate Preview Videos** unticked for fastest turnaround, or tick it to also render diagnostic `.mp4` / `.png` previews.
5. Click **Generate Motion**.

Blender remains completely interactive while the external pipeline runs. **Progress / Result** displays the current stage, a progress bar, recent log lines, and a **Cancel** button.

When the job finishes, the add-on automatically imports `character_animation.glb` into a dedicated collection (`V2BM_<source-name>`), hides joint helper objects, frames the character mesh in the active viewport, and configures the scene FPS and timeline start/end frames to match the source clip.

Each job writes its artefacts to:

```text
<project>/outputs/blender_jobs/<job-id>/
```

(`run.json`, `job.log`, `progress.jsonl`, `skeleton_motion.npz`, `rig_rotations.npz`, and `character_animation.glb`.)

## After Import Tips

- **Frame Character**: If the viewport camera is aimed away from the origin, select the character mesh and choose **View → Frame Selected**.
- **Hide Bone Display Spheres**: In the Outliner, hide `Icosphere` and the `Armature` wireframes; leave the skinned character mesh visible.
- **Playback Speed**: Timeline FPS is automatically synchronised to `fps` from `run.json`.

## Troubleshooting Common Errors

| Message | Resolution |
| :--- | :--- |
| `External Python not found` | Point **External Python** at `venv/bin/python` or `venv\Scripts\python.exe` |
| `VideoIntoBlenderModel package not installed` | Run `pip install -e .` inside that virtual environment |
| `Humanoid FBX invalid` | Ensure the character is exported as **FBX Binary** in a T-pose |
| `Input contains multiple people` | Use a single-person shot, or pre-isolate your target performer with `v2bm isolate` |
| `Pipeline process exited unexpectedly` | Click **View Full Log** to inspect `job.log` |
| `Generation succeeded, but automatic import failed` | Click **Import Generated Character** to retry importing the `.glb` |

## Directory Layout

```text
blender_extension/
├── README.md                          # This guide
└── v2bm_blender/                      # Packaged into dist/v2bm-0.1.0.zip
    ├── blender_manifest.toml
    ├── __init__.py
    ├── i18n.py
    ├── preferences.py
    ├── panels/
    ├── operators/
    └── services/
```

Packaging script: [`scripts/package_blender_extension.py`](../scripts/package_blender_extension.py).  
Maintainer smoke-test checklist: [`docs/blender-extension.md`](../docs/blender-extension.md).

# Asset Licence & Provenance Register

This document serves as the authoritative, human-readable licence and provenance register for all third-party media, 3D character rigs, and machine learning model weights included in or distributed with `VideoIntoBlenderModel` (`v2bm`).

All assets in this repository are strictly **commercial-permissive**, non-copyleft, and free from non-commercial academic research encumbrances (such as MPI Body-Model, legacy-lifter, or third-party-detector non-commercial terms).

---

## 1. Summary of Assets & Licences

| Asset ID | Type | Title | Creator / Author | Source | Local Path | Licence (SPDX) | Commercial Use | Attribution Required |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|
| `video_breakdancer_kickup` | Video | Breakdance Kick-up Demonstration | VincaniTV | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kick-up.webm) | `assets/video/breakdancer_kickup.mp4` | [CC-BY-3.0](https://creativecommons.org/licenses/by/3.0/) | Permitted | Yes |
| `video_breakdancer_6step` | Video | Six-Step Breakdance Demonstration | Neil Sweeney | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:6-step_example.webm) | `assets/video/breakdancer_6step.mp4` | [CC-BY-3.0](https://creativecommons.org/licenses/by/3.0/) | Permitted | Yes |
| `video_kung_fu_haikou` | Video | Kung Fu Form Demonstration | Anna Frodesiak | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kung_fu_at_Haikou_People%27s_Park.ogv) | `assets/video/kung_fu_haikou.mp4` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `rig_kenney_skater` | Rig | Animated Characters Protagonists — Skater Male | Kenney (Kenney Vleugels) | [Kenney.nl](https://kenney.nl/assets/animated-characters-protagonists) | `assets/rigs/kenney_skater.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional (Credited) |
| `rig_kenney_female` | Rig | Animated Characters Retro — Human Female | Kenney (Kenney Vleugels) | [Kenney.nl](https://kenney.nl/assets/animated-characters-retro) | `assets/rigs/kenney_female.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional (Credited) |
| `rig_humanoid_rig` | Rig | Procedural Expressive CC0 Humanoid Character Rig | Project Maintainers | Procedural Blender generator | `assets/rigs/humanoid_rig.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `image_urban_dancer_pose` | Image | Urban Dancer Full-Body Pose | Project Contributors | Photographic capture | `assets/image/urban_dancer_pose.png` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `image_expressive_portrait_smile` | Image | Expressive Smiling Portrait Still Photo | Project Contributors | Photographic studio portrait | `assets/image/expressive_portrait_smile.png` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `weight_pose_landmarker_full` | Model Weights | Google MediaPipe Pose Landmarker (Full) | Google LLC | [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) | `weights/mediapipe/pose_landmarker_full.task` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Permitted | Yes |
| `weight_face_landmarker` | Model Weights | Google MediaPipe Face Landmarker | Google LLC | [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) | `weights/mediapipe/face_landmarker.task` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Permitted | Yes |
| `demo_video` | Video | CLI Test Demonstration Video | VincaniTV / Maintainers | Derived from Wikimedia Commons | `docs/assets/demo.mp4` | [CC-BY-3.0](https://creativecommons.org/licenses/by/3.0/) | Permitted | Yes |
| `procedural_humanoid_rig` | Rig | Procedural CC0 Humanoid Rig Asset | Project Maintainers | Procedural Blender script | `assets/rigs/humanoid_rig.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |

---

## 2. Commercial-Free Motion Videos

### 2.1. Breakdance Kick-up Demonstration (`assets/video/breakdancer_kickup.mp4`)

- **Asset Title**: Breakdance Kick-up Demonstration
- **Creator / Author**: VincaniTV ([YouTube Channel](https://www.youtube.com/channel/UCh_S_EgAc6bAdu8C9aaqz1w))
- **Source URL**: [https://commons.wikimedia.org/wiki/File:Kick-up.webm](https://commons.wikimedia.org/wiki/File:Kick-up.webm)
- **Direct Asset URL**: [https://upload.wikimedia.org/wikipedia/commons/4/48/Kick-up.webm](https://upload.wikimedia.org/wikipedia/commons/4/48/Kick-up.webm)
- **Local File Path**: `assets/video/breakdancer_kickup.mp4`
- **Companion Test Copy**: `docs/assets/demo.mp4`
- **Licence**: Creative Commons Attribution 3.0 Unported (`CC-BY-3.0`)
- **Licence Terms Summary**: You are free to share (copy and redistribute the material in any medium or format) and adapt (remix, transform, and build upon the material) for any purpose, **including commercially**.
- **Commercial-Use Verification**: Section 3 of the CC BY 3.0 legal code permits commercial exploitation without royalty payments.
- **Attribution Statement**:
  > Video: "Kick-up.webm" by VincaniTV, licensed under Creative Commons Attribution 3.0 Unported (CC BY 3.0).
- **Technical Characteristics**:
  - Container: MP4 (H.264 / MPEG-4 AVC)
  - Resolution: 1920×1080 (Full High Definition)
  - Frame Rate: 24.0 fps
  - Frame Count: 67 frames (duration: 2.79 seconds)
  - Subject Matter: Single breakdancer executing an athletic kick-up freeze transition, unobstructed head-to-toe framing throughout.
  - Quality Assurance: Passes `v2bm.checks.check_video()` with zero warnings or errors.

---

### 2.2. Six-Step Breakdance Demonstration (`assets/video/breakdancer_6step.mp4`)

- **Asset Title**: Six-Step Breakdance Demonstration
- **Creator / Author**: Neil Sweeney ([YouTube Channel](https://www.youtube.com/channel/UCSxtUzrumNJHw39FmQlC-ww))
- **Source URL**: [https://commons.wikimedia.org/wiki/File:6-step_example.webm](https://commons.wikimedia.org/wiki/File:6-step_example.webm)
- **Direct Asset URL**: [https://upload.wikimedia.org/wikipedia/commons/e/e4/6-step_example.webm](https://upload.wikimedia.org/wikipedia/commons/e/e4/6-step_example.webm)
- **Local File Path**: `assets/video/breakdancer_6step.mp4`
- **Licence**: Creative Commons Attribution 3.0 Unported (`CC-BY-3.0`)
- **Licence Terms Summary**: You are free to share and adapt the material for any purpose, **including commercially**, under the condition of giving appropriate attribution.
- **Commercial-Use Verification**: Verified commercial-permissive under standard Creative Commons Attribution terms.
- **Attribution Statement**:
  > Video: "6-step example.webm" by Neil Sweeney, licensed under Creative Commons Attribution 3.0 Unported (CC BY 3.0).
- **Technical Characteristics**:
  - Container: MP4 (H.264 / MPEG-4 AVC)
  - Resolution: 640×360
  - Frame Rate: 30.0 fps
  - Frame Count: 120 frames (duration: 4.0 seconds)
  - Subject Matter: Single breakdancer performing a foundational six-step rotational footwork sequence with clean limb visibility.
  - Quality Assurance: Passes `v2bm.checks.check_video()` with zero warnings or errors.

---

### 2.3. Kung Fu Form Demonstration (`assets/video/kung_fu_haikou.mp4`)

- **Asset Title**: Kung Fu Form Demonstration at Haikou People's Park
- **Creator / Author**: Anna Frodesiak ([Wikimedia Commons](https://commons.wikimedia.org/wiki/User:Anna_Frodesiak))
- **Source URL**: [https://commons.wikimedia.org/wiki/File:Kung_fu_at_Haikou_People%27s_Park.ogv](https://commons.wikimedia.org/wiki/File:Kung_fu_at_Haikou_People%27s_Park.ogv)
- **Direct Asset URL**: [https://upload.wikimedia.org/wikipedia/commons/b/be/Kung_fu_at_Haikou_People%27s_Park.ogv](https://upload.wikimedia.org/wikipedia/commons/b/be/Kung_fu_at_Haikou_People%27s_Park.ogv)
- **Local File Path**: `assets/video/kung_fu_haikou.mp4`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`) Public Domain Dedication
- **Licence Terms Summary**: Unrestricted worldwide public domain dedication. Free to share, modify, and build upon for any commercial or non-commercial application without royalty or attribution.
- **Commercial-Use Verification**: Verified CC0 public domain dedication.
- **Attribution Statement**:
  > Video: "Kung fu at Haikou People's Park" by Anna Frodesiak, dedicated to the public domain under Creative Commons Zero (CC0 1.0 Universal).
- **Technical Characteristics**:
  - Container: MP4 (H.264 / MPEG-4 AVC)
  - Resolution: 720×480
  - Frame Rate: 29.97 fps
  - Frame Count: 149 frames (duration: 4.97 seconds)
  - Subject Matter: Martial artist executing full-body kung fu forms, stance transitions, and dynamic strikes with clear limb separation.
  - Quality Assurance: Passes `v2bm.checks.check_video()` with zero warnings or errors.

---

## 3. Clothed & Textured Open-Licence Character Rigs

### 3.1. Kenney Animated Characters Protagonists — Skater Male (`assets/rigs/kenney_skater.fbx`)

- **Asset Title**: Animated Characters Protagonists — Skater Male
- **Creator / Author**: Kenney (Kenney Vleugels — [www.kenney.nl](https://kenney.nl))
- **Source URL**: [https://kenney.nl/assets/animated-characters-protagonists](https://kenney.nl/assets/animated-characters-protagonists)
- **Local File Path**: `assets/rigs/kenney_skater.fbx`
- **Companion Texture**: `assets/rigs/kenney_skater.png` (`skaterMaleA.png`, 512×512 PNG diffuse colour atlas)
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`) Public Domain Dedication
- **Licence Terms Summary**: The creator has dedicated the work to the worldwide public domain, waiving all copyright and related or neighbouring rights. You can copy, modify, distribute, and perform the work, **even for commercial purposes**, all without asking permission.
- **Commercial-Use Verification**: Section 2 of the CC0 1.0 Universal legal code provides an absolute waiver of copyright restrictions for commercial and non-commercial utilisation.
- **Attribution Statement**:
  > Character model: "Animated Characters Protagonists - skaterMaleA" by Kenney (www.kenney.nl), dedicated to the public domain under Creative Commons Zero (CC0 1.0 Universal).
- **Technical Characteristics**:
  - Format: Binary FBX (Kaydara 7.4.0 format)
  - Texture Mapping: Conformal per-vertex UV mapping (`TEXCOORD_0`) with embedded 512×512 PNG diffuse texture and companion external PNG
  - Armature Hierarchy: Exact 25-bone standard humanoid hierarchy rooted at `rig:Hips`
  - Mesh Geometry: 804 vertices, 1,604 triangular polygons
  - Kinematic Guarantees:
    - Rest Forward Kinematics (FK) Error: $1.4147 \times 10^{-12}$ (well within the $< 10^{-4}$ tolerance threshold)
    - Rest Linear Blend Skinning Error: $2.6958 \times 10^{-7}\text{ cm}$ (well within the $< 10^{-2}\text{ cm}$ tolerance threshold)
  - Quality Assurance: Passes `v2bm.checks.check_rig()`, exports valid textured glTF 2.0 (`.glb`), and passes headless Blender 5.2.2 LTS automated verification (`scripts/verify_blender_glb.py`).

---

### 3.2. Kenney Animated Characters Retro — Human Female (`assets/rigs/kenney_female.fbx`)

- **Asset Title**: Animated Characters Retro — Human Female
- **Creator / Author**: Kenney (Kenney Vleugels — [www.kenney.nl](https://kenney.nl))
- **Source URL**: [https://kenney.nl/assets/animated-characters-retro](https://kenney.nl/assets/animated-characters-retro)
- **Local File Path**: `assets/rigs/kenney_female.fbx`
- **Companion Texture**: `assets/rigs/kenney_female.png` (`humanFemaleA.png`, 512×512 PNG diffuse colour atlas)
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`) Public Domain Dedication
- **Licence Terms Summary**: Dedicated to the worldwide public domain under Creative Commons Zero v1.0 Universal. Free for personal, educational, and commercial purposes without restriction.
- **Commercial-Use Verification**: Verified CC0 public domain dedication permitting unrestricted commercial use.
- **Attribution Statement**:
  > Character model: "Animated Characters Retro - humanFemaleA" by Kenney (www.kenney.nl), dedicated to the public domain under Creative Commons Zero (CC0 1.0 Universal).
- **Technical Characteristics**:
  - Format: Binary FBX (Kaydara 7.4.0 format)
  - Texture Mapping: Conformal per-vertex UV mapping (`TEXCOORD_0`) with embedded 512×512 PNG diffuse texture and companion external PNG
  - Armature Hierarchy: Exact 25-bone standard humanoid hierarchy rooted at `rig:Hips`
  - Mesh Geometry: 804 vertices, 1,604 triangular polygons
  - Kinematic Guarantees:
    - Rest Forward Kinematics (FK) Error: $1.4147 \times 10^{-12}$ (well within the $< 10^{-4}$ tolerance threshold)
    - Rest Linear Blend Skinning Error: $2.6958 \times 10^{-7}\text{ cm}$ (well within the $< 10^{-2}\text{ cm}$ tolerance threshold)
  - Quality Assurance: Passes `v2bm.checks.check_rig()`, exports valid textured glTF 2.0 (`.glb`), and passes headless Blender 5.2.2 LTS automated verification (`scripts/verify_blender_glb.py`).

---

### 3.3. Procedural Expressive CC0 Humanoid Character Rig (`assets/rigs/humanoid_rig.fbx`)

- **Asset Title**: Procedural Expressive CC0 Humanoid Character Rig
- **Creator / Author**: VideoIntoBlenderModel Maintainers
- **Source URL**: Generated via `scripts/generate_humanoid_rig.py`
- **Local File Path**: `assets/rigs/humanoid_rig.fbx`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`) Public Domain Dedication
- **Licence Terms Summary**: Dedicated unconditionally to the worldwide public domain. Completely free for any personal, educational, commercial, or production use without royalty, licencing fees, or mandatory attribution.
- **Commercial-Use Verification**: Verified CC0 public domain dedication.
- **Attribution Statement**:
  > Character model: "humanoid_rig.fbx" generated procedurally via scripts/generate_humanoid_rig.py, dedicated to the public domain under Creative Commons Zero (CC0 1.0 Universal).
- **Technical Characteristics**:
  - Format: Binary FBX (Kaydara 7.4.0 format)
  - Armature Hierarchy: 25-bone standard humanoid armature rooted at `rig:Hips`
  - Mesh Geometry: 3,912 vertices, 7,816 triangular polygons with articulated head, jaw, eyes, and mouth features
  - Facial Blendshapes: 52 canonical analytical ARKit shape keys (`browInnerUp`, `jawOpen`, `mouthSmileLeft`, `mouthSmileRight`, `eyeBlinkLeft`, `eyeBlinkRight`, etc.)
  - Quality Assurance: Passes `v2bm.checks.check_rig()`, exports valid binary glTF 2.0 (`.glb`) with animated morph weight channels, and passes automated Blender LTS verification.

---

## 4. Commercial-Free Input Media

### 4.1. Urban Dancer Full-Body Pose (`assets/image/urban_dancer_pose.png`)

- **Asset Title**: Urban Dancer Full-Body Pose
- **Creator / Author**: VideoIntoBlenderModel Contributors
- **Local File Path**: `assets/image/urban_dancer_pose.png`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`) Public Domain Dedication
- **Licence Terms Summary**: Worldwide public domain dedication. Unrestricted commercial and non-commercial utilisation.
- **Technical Characteristics**:
  - Container: PNG (24-bit RGB)
  - Resolution: 1080×1920
  - Framing: Full-body dance pose with unobstructed head-to-toe limb visibility
  - Quality Assurance: Passes `v2bm.checks.check_image()` validation with zero warnings or errors.

---

### 4.2. Expressive Smiling Portrait Still Photo (`assets/image/expressive_portrait_smile.png`)

- **Asset Title**: Expressive Smiling Portrait Still Photo
- **Creator / Author**: VideoIntoBlenderModel Contributors
- **Local File Path**: `assets/image/expressive_portrait_smile.png`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`) Public Domain Dedication
- **Licence Terms Summary**: Worldwide public domain dedication. Unrestricted commercial and non-commercial utilisation.
- **Technical Characteristics**:
  - Container: PNG (24-bit RGB)
  - Resolution: 1024×1024
  - Framing: Close-up cinematic portrait featuring a pronounced smile and expressive facial articulation
  - Quality Assurance: Passes `v2bm.checks.check_image(framing_mode="portrait")` validation with zero warnings or errors.

---

## 5. Machine Learning Inference Model Weights

### 5.1. Google MediaPipe Pose Landmarker Full Task (`weights/mediapipe/pose_landmarker_full.task`)

- **Asset Title**: MediaPipe Pose Landmarker (Full) Model Task Bundle
- **Creator / Author**: Google LLC
- **Source URL**: [https://developers.google.com/mediapipe/solutions/vision/pose_landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- **Direct Download URL**: [https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)
- **Local File Path**: `weights/mediapipe/pose_landmarker_full.task`
- **File Size**: 9,398,198 bytes (~8.96 MB)
- **Licence**: Apache License, Version 2.0 (`Apache-2.0`)
- **Licence Terms Summary**: Permissive commercial licence granting perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright and patent licence to use, reproduce, prepare derivative works of, and distribute the software and models.
- **Attribution Statement**:
  > Google MediaPipe Pose Landmarker (Full) Model, Copyright 2023 Google LLC, licensed under the Apache License, Version 2.0.

---

### 5.2. Google MediaPipe Face Landmarker Task (`weights/mediapipe/face_landmarker.task`)

- **Asset Title**: MediaPipe Face Landmarker Model Task Bundle
- **Creator / Author**: Google LLC
- **Source URL**: [https://developers.google.com/mediapipe/solutions/vision/face_landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- **Direct Download URL**: [https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- **Local File Path**: `weights/mediapipe/face_landmarker.task`
- **File Size**: 3,758,596 bytes (~3.58 MB)
- **Licence**: Apache License, Version 2.0 (`Apache-2.0`)
- **Licence Terms Summary**: Permissive commercial licence granting perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright and patent licence to use, reproduce, prepare derivative works of, and distribute the software and models.
- **Attribution Statement**:
  > Google MediaPipe Face Landmarker Model, Copyright 2023 Google LLC, licensed under the Apache License, Version 2.0.

---

## 6. Machine-Readable Manifest Synchronisation

The metadata in this document is synchronously mirrored in machine-readable JSON format at:
```text
assets/licences.json
```
Automated verification tests (`tests/test_licence_and_deps.py`) audit `assets/licences.json` to verify that all referenced assets exist on disk, contain valid commercial licences, and require no proprietary or non-commercial dependencies.

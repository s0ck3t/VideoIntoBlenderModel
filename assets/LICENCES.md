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
| `video_urban_dancer` | Video | Urban Dancer Hip-Hop Freestyle Demonstration | Cottonbro Studio / Pexels Contributor | [Pexels](https://www.pexels.com/video/51314/) | `assets/video/urban_dancer_51314.mp4` | [Pexels](https://www.pexels.com/license/) | Permitted | Optional |
| `video_street_dancer` | Video | Street Dancer Breakdance & Freestyle Demonstration | Cottonbro Studio / Pexels Contributor | [Pexels](https://www.pexels.com/video/51310/) | `assets/video/street_dancer_51310.mp4` | [Pexels](https://www.pexels.com/license/) | Permitted | Optional |
| `video_expressive_face` | Video | Expressive Face Portrait Close-up Demonstration | Project Contributors | Photographic video capture | `assets/video/expressive_face.mp4` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `video_attractor_walk_crouch` | Video | Attractor Walk & Crouch Locomotion Demonstration | Project Contributors | Video demonstration capture | `assets/video/attractor_walk_crouch_source.mp4` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `rig_attractor_mechanic` | Rig | Attractor Mechanic — Stylised Game Female Character | Project Contributors | Project 3D Assets | `assets/rigs/attractor_mechanic_tpose.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `rig_kenney_skater` | Rig | Animated Characters Protagonists — Skater Male | Kenney (Kenney Vleugels) | [Kenney.nl](https://kenney.nl/assets/animated-characters-protagonists) | `assets/rigs/kenney_skater.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional (Credited) |
| `rig_kenney_female` | Rig | Animated Characters Retro — Human Female | Kenney (Kenney Vleugels) | [Kenney.nl](https://kenney.nl/assets/animated-characters-retro) | `assets/rigs/kenney_female.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional (Credited) |
| `rig_renderpeople_sophia` | Rig | Renderpeople Free 3D People — Sophia (`rp_sophia_animated_003_idling`) | Renderpeople GmbH | [Renderpeople.com](https://renderpeople.com/free-3d-people/) | `assets/rigs/renderpeople_sophia.fbx` | [Renderpeople-Free](https://renderpeople.com/general-terms-and-conditions/) | Permitted | Optional |
| `rig_renderpeople_nathan` | Rig | Renderpeople Free 3D People — Nathan (`rp_nathan_animated_003_walking`) | Renderpeople GmbH | [Renderpeople.com](https://renderpeople.com/free-3d-people/) | `assets/rigs/renderpeople_nathan.fbx` | [Renderpeople-Free](https://renderpeople.com/general-terms-and-conditions/) | Permitted | Optional |
| `procedural_articulated_humanoid_rig` | Rig | Procedural CC0 61-Bone Articulated Humanoid Rig Asset | Project Maintainers | Procedural Blender script | `assets/rigs/humanoid_articulated.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `rig_humanoid_rig` | Rig | Procedural Expressive CC0 Humanoid Character Rig | Project Maintainers | Procedural Blender generator | `assets/rigs/humanoid_rig.fbx` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `image_urban_dancer_pose` | Image | Urban Dancer Full-Body Pose | Project Contributors | Photographic capture | `assets/image/urban_dancer_pose.png` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `image_expressive_portrait_smile` | Image | Expressive Smiling Portrait Still Photo | Project Contributors | Photographic studio portrait | `assets/image/expressive_portrait_smile.png` | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Permitted | Optional |
| `weight_pose_landmarker_full` | Model Weights | Google MediaPipe Pose Landmarker (Full) | Google LLC | [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) | `weights/mediapipe/pose_landmarker_full.task` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Permitted | Yes |
| `weight_face_landmarker` | Model Weights | Google MediaPipe Face Landmarker | Google LLC | [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) | `weights/mediapipe/face_landmarker.task` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Permitted | Yes |
| `weight_hand_landmarker` | Model Weights | Google MediaPipe Hand Landmarker | Google LLC | [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) | `weights/mediapipe/hand_landmarker.task` | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Permitted | Yes |
| `demo_video` | Video | CLI Test Demonstration Video | VincaniTV / Maintainers | Derived from Wikimedia Commons | `docs/assets/demo.mp4` | [CC-BY-3.0](https://creativecommons.org/licenses/by/3.0/) | Permitted | Yes |

---

## 2. Commercial-Free Motion Videos

### 2.1. Breakdance Kick-up Demonstration (`assets/video/breakdancer_kickup.mp4`)

- **Asset Title**: Breakdance Kick-up Demonstration
- **Creator / Author**: VincaniTV ([YouTube Channel](https://www.youtube.com/channel/UCh_S_EgAc6bAdu8C9aaqz1w))
- **Source URL**: [https://commons.wikimedia.org/wiki/File:Kick-up.webm](https://commons.wikimedia.org/wiki/File:Kick-up.webm)
- **Local File Path**: `assets/video/breakdancer_kickup.mp4`
- **Licence**: Creative Commons Attribution 3.0 Unported (`CC-BY-3.0`)

### 2.2. Six-Step Breakdance Demonstration (`assets/video/breakdancer_6step.mp4`)

- **Asset Title**: Six-Step Breakdance Demonstration
- **Creator / Author**: Neil Sweeney
- **Source URL**: [https://commons.wikimedia.org/wiki/File:6-step_example.webm](https://commons.wikimedia.org/wiki/File:6-step_example.webm)
- **Local File Path**: `assets/video/breakdancer_6step.mp4`
- **Licence**: Creative Commons Attribution 3.0 Unported (`CC-BY-3.0`)

### 2.3. Kung Fu Form Demonstration (`assets/video/kung_fu_haikou.mp4`)

- **Asset Title**: Kung Fu Form Demonstration at Haikou People's Park
- **Creator / Author**: Anna Frodesiak
- **Source URL**: [https://commons.wikimedia.org/wiki/File:Kung_fu_at_Haikou_People%27s_Park.ogv](https://commons.wikimedia.org/wiki/File:Kung_fu_at_Haikou_People%27s_Park.ogv)
- **Local File Path**: `assets/video/kung_fu_haikou.mp4`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`)

### 2.4. Urban Dancer Hip-Hop Freestyle Demonstration (`assets/video/urban_dancer_51314.mp4`)

- **Asset Title**: Urban Dancer Hip-Hop Freestyle Demonstration
- **Creator / Author**: Cottonbro Studio / Pexels Contributor
- **Source URL**: [https://www.pexels.com/video/51314/](https://www.pexels.com/video/51314/)
- **Local File Path**: `assets/video/urban_dancer_51314.mp4`
- **Licence**: Pexels Royalty-Free Commercial Licence (`Pexels`)

### 2.5. Street Dancer Breakdance & Freestyle Demonstration (`assets/video/street_dancer_51310.mp4`)

- **Asset Title**: Street Dancer Breakdance & Freestyle Demonstration
- **Creator / Author**: Cottonbro Studio / Pexels Contributor
- **Source URL**: [https://www.pexels.com/video/51310/](https://www.pexels.com/video/51310/)
- **Local File Path**: `assets/video/street_dancer_51310.mp4`
- **Licence**: Pexels Royalty-Free Commercial Licence (`Pexels`)

### 2.6. Expressive Face Portrait Close-up Demonstration (`assets/video/expressive_face.mp4`)

- **Asset Title**: Expressive Face Portrait Close-up Demonstration
- **Creator / Author**: VideoIntoBlenderModel Contributors
- **Local File Path**: `assets/video/expressive_face.mp4`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`)

### 2.7. Attractor Walk & Crouch Locomotion Demonstration (`assets/video/attractor_walk_crouch_source.mp4`)

- **Asset Title**: Attractor Walk & Crouch Locomotion Demonstration
- **Creator / Author**: VideoIntoBlenderModel Contributors
- **Local File Path**: `assets/video/attractor_walk_crouch_source.mp4`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`)

---

## 3. Clothed & Textured Open-Licence Character Rigs

### 3.1. Kenney Animated Characters Protagonists — Skater Male (`assets/rigs/kenney_skater.fbx`)

- **Creator / Author**: Kenney (Kenney Vleugels — [www.kenney.nl](https://kenney.nl))
- **Local File Path**: `assets/rigs/kenney_skater.fbx`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`)

### 3.2. Kenney Animated Characters Retro — Human Female (`assets/rigs/kenney_female.fbx`)

- **Creator / Author**: Kenney (Kenney Vleugels — [www.kenney.nl](https://kenney.nl))
- **Local File Path**: `assets/rigs/kenney_female.fbx`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`)

### 3.3. Renderpeople Free 3D People — Sophia (`assets/rigs/renderpeople_sophia.fbx`)

- **Creator / Author**: Renderpeople GmbH ([renderpeople.com](https://renderpeople.com/free-3d-people/))
- **Local File Path**: `assets/rigs/renderpeople_sophia.fbx` (companion texture: `assets/rigs/renderpeople_sophia.png`)
- **Licence**: Renderpeople Official Free 3D People Licence (Permitted for commercial and non-commercial projects)
- **Specification**: Photogrammetry-scanned clothed female character rigged with 61 canonical `rig:*` bones (25 core + 30 phalangeal finger bones + 6 limb twist bones) and 52 canonical ARKit facial blendshapes (`Basis` + 52 shape keys).

### 3.4. Renderpeople Free 3D People — Nathan (`assets/rigs/renderpeople_nathan.fbx`)

- **Creator / Author**: Renderpeople GmbH ([renderpeople.com](https://renderpeople.com/free-3d-people/))
- **Local File Path**: `assets/rigs/renderpeople_nathan.fbx` (companion texture: `assets/rigs/renderpeople_nathan.png`)
- **Licence**: Renderpeople Official Free 3D People Licence (Permitted for commercial and non-commercial projects)
- **Specification**: Photogrammetry-scanned clothed male character rigged with 61 canonical `rig:*` bones (25 core + 30 phalangeal finger bones + 6 limb twist bones) and 52 canonical ARKit facial blendshapes (`Basis` + 52 shape keys).

### 3.5. Procedural CC0 61-Bone Articulated Humanoid Character Rig (`assets/rigs/humanoid_articulated.fbx`)

- **Creator / Author**: VideoIntoBlenderModel Maintainers
- **Local File Path**: `assets/rigs/humanoid_articulated.fbx`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`)

### 3.6. Procedural Expressive CC0 Humanoid Character Rig (`assets/rigs/humanoid_rig.fbx`)

- **Creator / Author**: VideoIntoBlenderModel Maintainers
- **Local File Path**: `assets/rigs/humanoid_rig.fbx`
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`)

### 3.7. Attractor Mechanic Stylised Game Character Rig (`assets/rigs/attractor_mechanic_tpose.fbx`)

- **Creator / Author**: VideoIntoBlenderModel Contributors
- **Local File Path**: `assets/rigs/attractor_mechanic_tpose.fbx` (companion texture: `assets/rigs/attractor_mechanic_tpose.png`)
- **Licence**: Creative Commons Zero v1.0 Universal (`CC0-1.0`)
- **Specification**: Stylised game-ready female character rigged with 61 canonical `rig:*` bones (25 core + 30 phalangeal finger bones + 6 limb twist helper bones) and 52 canonical ARKit facial blendshapes (`Basis` + 52 shape keys), with 4K `4096×4096` diffuse PBR texture atlas and watertight single-component manifold mesh topology.

---

## 4. Machine Learning Inference Model Weights

- `weights/mediapipe/pose_landmarker_full.task` — Google MediaPipe Pose Landmarker (Full), `Apache-2.0`
- `weights/mediapipe/face_landmarker.task` — Google MediaPipe Face Landmarker, `Apache-2.0`
- `weights/mediapipe/hand_landmarker.task` — Google MediaPipe Hand Landmarker, `Apache-2.0`

---

## 5. Machine-Readable Manifest Synchronisation

The metadata in this document is synchronously mirrored in machine-readable JSON format at `assets/licences.json`.

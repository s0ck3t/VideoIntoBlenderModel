# TEST_INFRA.md: VideoIntoBlenderModel Test Architecture & Verification Infrastructure

## 1. Executive Summary

This document establishes the authoritative testing architecture, mathematical verification derivations, and execution standards for the commercial-free reproduction of **VideoIntoBlenderModel**.

The test infrastructure enforces strict quality gates across four sequential tiers:
- **Tier 1: Algorithmic & Licence Guarantees** — Verifies 100% permissive licensing (Apache-2.0, MIT, BSD, CC0) and total absence of copyleft (AGPL) or non-commercial research models (MPI Body-Model, legacy-lifter, third-party-detector YOLOv8, and legacy legacy_mesh_serializer).
- **Tier 2: Mathematical Verification** — Mathematical proofs of machine-precision forward kinematics (`rest_fk_error` $\le 10^{-12}$), sub-millimetre linear blend skinning (`rest_skin_error` $< 10^{-3}\text{ cm}$), retargeting continuity (`twist_events = 0`, `quat_sign_flips = 0`), and adaptive One-Euro (1€) filter acceleration noise / jerk reduction ($\ge 60\%$).
- **Tier 3: End-to-End CLI Verification** — Opaque-box execution testing for `v2bm doctor`, `v2bm run --video`, and `v2bm run --image`.
- **Tier 4: Headless Blender 5.2.2 LTS Verification** — Headless automated 3D scene audit and rendering harness verifying armature hierarchy, skinned meshes, keyframed actions, and visual fidelity.

---

## 2. Directory Layout & Test Suite Organisation

```
VideoIntoBlenderModel/
├── pyproject.toml                         # Permissive dependencies (MediaPipe, OpenCV, NumPy, SciPy)
├── v2bm/                                  # Core pipeline implementation
│   ├── checks.py                          # MediaPipe pose preflight & validation
│   ├── cli.py                             # CLI commands: doctor, run, isolate
│   ├── extract.py                         # 3D landmark extraction, 1€ filter, biomechanics
│   ├── human_motion.py                    # HumanMotion data container & loader
│   ├── isolate.py                         # Multi-person subject isolation & tracking
│   └── rig/                               # 100% Permissive analytical retargeting & glTF exporter
│       ├── animation.py                   # RetargetedAnimation & frame-by-frame orchestration
│       ├── fbx_skeleton.py                # Binary FBX parser & bone hierarchy extractor
│       ├── gltf.py                        # Pure Python binary GLB exporter with skeletal keyframes
│       ├── kinematics.py                  # FK, quaternion continuity, rotationAligning
│       ├── retarget.py                    # Analytical swing-twist solver & aim twist continuity
│       ├── skinned_mesh.py                # FBX geometry, cluster, IBM, & linear blend skinning
│       ├── tpose_calibration.py           # Anatomical axis measurement & rig calibration
│       └── rig_retarget.py                # Rest-relative motion mapping & Humanoid Rig FK
├── scripts/
│   ├── generate_humanoid_rig.py           # Procedural CC0 character generator (Blender 5.2.2 LTS)
│   ├── package_blender_extension.py       # Extension packager
│   └── verify_blender_glb.py              # Tier 4 Headless Blender GLB verification script
└── tests/
    ├── __init__.py                        # Test suite package initialisation
    ├── test_licence_and_deps.py           # Tier 1: Licence, AST imports, and asset compliance
    ├── test_fk_skin.py                    # Tier 2: Forward kinematics & linear blend skinning maths
    ├── test_retarget_stability.py         # Tier 2: Retargeting twist stability & quaternion continuity
    ├── test_one_euro_filter.py            # Tier 2: 1€ Filter jerk & acceleration noise reduction
    └── test_cli.py                        # Tier 3: CLI opaque-box end-to-end execution
```

---

## 3. Authoritative Expected Output Derivations

### 3.1. Tier 1: Licence & Dependency Guarantees (`tests/test_licence_and_deps.py`)

#### Prohibited Dependencies & Models Blacklist
- **MPI Body-Model (`bodymodel`, `neutral_body_model.npz`)**: Non-commercial academic research licence. Strictly prohibited from repository manifests, AST imports, and assets.
- **legacy-lifter (`legacy_lifter`)**: Custom non-commercial research licence + proprietary 5 GB weight checkpoints. Strictly prohibited.
- **third-party-detector YOLOv8 (`third_party_detector`)**: AGPL-3.0 copyleft licence. Strictly prohibited.
- **Legacy LegacyMeshSerializer (`legacy_mesh_serializer`)**: Obsolete build-isolation breaking dependency. Strictly prohibited.

#### Permissive Ecosystem Whitelist
All runtime and build dependencies declared in `pyproject.toml` must belong to the permissive whitelist:
`{numpy, opencv-python, mediapipe, scipy, pytest, einops, setuptools}` (Apache-2.0, MIT, BSD).

#### Source Tree AST Inspection
An abstract syntax tree (AST) traversal over all Python files under `v2bm/` asserts that for every file $f \in \text{SourceFiles}$ and every import $\text{node} \in \operatorname{AST}(f)$, the root package name is strictly disjoint from `{bodymodel, legacy_lifter, third_party_detector, legacy_mesh_serializer}`.

---

### 3.2. Tier 2: Mathematical Verification

#### 3.2.1. Forward Kinematics Machine Precision (`tests/test_fk_skin.py`)
In the hierarchical skeletal model, the world transformation matrix $\mathbf{M}_i \in \mathrm{SE}(3)$ for bone $i$ with parent $p(i)$ is:
$$\mathbf{M}_i = \mathbf{M}_{p(i)} \cdot \mathbf{T}_i \cdot \mathbf{R}_{\text{pre}, i} \cdot \mathbf{R}_{\text{anim}, i}$$
At rest pose, $\mathbf{R}_{\text{anim}, i} = \mathbf{I}_{3\times3}$. When edit bones are oriented uniformly along $+Y$, $\mathbf{R}_{\text{pre}, i} = \mathbf{I}_{3\times3}$ and $\operatorname{Lcl\ Rotation}_i = (0, 0, 0)$.

Thus, forward kinematics simplifies to:
$$\mathbf{M}_i = \mathbf{M}_{p(i)} \cdot \mathbf{T}_i = \mathbf{M}_{i, \text{rest}}$$
The rest FK error metric (`rest_fk_error`) is defined as:
$$\epsilon_{\text{FK,rest}} = \max_{i \in \text{Bones}} \|\mathbf{M}_{i, \text{FK}}(\mathbf{I}) - \mathbf{M}_{i, \text{rest}}\|_F$$
**Authoritative Bound**: `rest_fk_error` $\le 10^{-12}$ (machine floating-point precision).

#### 3.2.2. Linear Blend Skinning Rest Error (`tests/test_fk_skin.py`)
Linear blend skinning (LBS) maps bind-pose vertex coordinates $\mathbf{v} \in \mathbb{R}^3$ to posed space $\mathbf{v}'$ via inverse bind matrices (IBM) $\mathbf{B}_j = \mathbf{M}_{j, \text{rest}}^{-1}$ and bone weights $w_{v, j}$:
$$\mathbf{v}' = \sum_{j} w_{v, j} \cdot (\mathbf{M}_j \mathbf{B}_j) \cdot \mathbf{v}$$
Subject to the partition of unity:
$$\sum_{j} w_{v, j} = 1.0, \quad w_{v, j} \ge 0$$
At rest pose ($\mathbf{M}_j = \mathbf{M}_{j, \text{rest}}$):
$$\mathbf{M}_j \mathbf{B}_j = \mathbf{M}_{j, \text{rest}} \mathbf{M}_{j, \text{rest}}^{-1} = \mathbf{I}_{4\times4} \implies \mathbf{v}' = \sum_j w_{v, j} \mathbf{I} \mathbf{v} = \mathbf{v}$$
The rest skinning error (`rest_skin_error`) is:
$$\epsilon_{\text{skin,rest}} = \max_{v \in \text{Vertices}} \|\mathbf{v}' - \mathbf{v}\|_2$$
**Authoritative Bound**: `rest_skin_error` $< 10^{-3}\text{ cm}$ (sub-micrometre threshold).

#### 3.2.3. Retargeting Twist Stability & Quaternion Geodesic Continuity (`tests/test_retarget_stability.py`)
The analytical retargeter determines bone orientations via swing-twist decomposition. For degrees of freedom with discrete $180^\circ$ twist ambiguity, `pick_closest_aim_twist` selects the variant minimising rotation distance to the previous frame's orientation:
$$\theta_{\text{twist}} = \arg\min_{\phi \in \{0, \pi\}} \|\mathbf{R}(\phi) - \mathbf{R}_{t-1}\|_F$$
Under smooth continuous joint trajectories, no unexpected discrete $180^\circ$ jumps occur (`twist_events = 0`):
$$N_{\text{twist}} = 0$$

To eliminate $C^0$ discontinuities in animated quaternion tracks caused by the $S^3 \to \mathrm{SO}(3)$ antipodal double cover $(\mathbf{q} \sim -\mathbf{q})$, `continue_quaternion` negates $\mathbf{q}_t$ whenever $\mathbf{q}_t \cdot \mathbf{q}_{t-1}^* < 0$. This guarantees the inner product remains non-negative across all keyframed frames (`quat_sign_flips = 0`):
$$\forall t \ge 1, \quad \mathbf{q}_t^* \cdot \mathbf{q}_{t-1}^* \ge 0 \implies N_{\text{flip}} = 0$$

#### 3.2.4. Adaptive One-Euro (1€) Filter Acceleration Noise Reduction (`tests/test_one_euro_filter.py`)
Monocular landmark depth coordinates exhibit high-frequency spatial jitter. The frame-to-frame jerk $\mathbf{j}(t) \in \mathbb{R}^3$ measures the third derivative (acceleration noise):
$$\mathbf{j}(t) = \frac{\mathbf{x}(t) - 3\mathbf{x}(t-1) + 3\mathbf{x}(t-2) - \mathbf{x}(t-3)}{T_e^3}$$
The mean squared jerk is:
$$\operatorname{MSJ} = \frac{1}{N-3} \sum_{t=3}^{N-1} \|\mathbf{j}(t)\|_2^2$$
The adaptive 1€ filter dynamically modulates its low-pass cutoff frequency based on instantaneous landmark velocity:
$$f_c = f_{c,\min} + \beta \|\hat{\dot{\mathbf{x}}}_t\|$$
With axis-specialised parameters:
- Lateral/Vertical ($X, Y$): $f_{c,\min} = 1.0\text{ Hz}$, $\beta = 0.005$, $f_{c,d} = 1.0\text{ Hz}$
- Depth ($Z$): $f_{c,\min} = 0.5\text{ Hz}$, $\beta = 0.05$, $f_{c,d} = 1.0\text{ Hz}$

**Authoritative Bound**:
$$\text{Jerk Reduction} = \frac{\operatorname{MSJ}_{\text{raw}} - \operatorname{MSJ}_{\text{filtered}}}{\operatorname{MSJ}_{\text{raw}}} \ge 60\% \quad (0.60)$$

---

### 3.3. Tier 3: End-to-End CLI Verification (`tests/test_cli.py`)

- **`v2bm doctor`**: Preflight verification checking assets, rigs, and dependencies.
  - Exit code: 0 when assets are satisfied (or code 1 with clean diagnostics).
  - Cleanliness: Stderr and stdout must contain zero mentions of `neutral_body_model.npz`, `bodymodel`, `legacy_lifter`, or `third_party_detector`.
- **`v2bm run --video <demo.mp4>`**:
  - Exit code: 0.
  - Output artifacts generated:
    - `skeleton_motion.npz` (shape `(F, 24, 3)`, metres, Y-up right-handed).
    - `rig_rotations.npz`.
    - `character_animation.glb` (binary glTF 2.0 with skeletal animation).
    - `videos/compare.mp4` (multi-view comparison render).
- **`v2bm run --image <repo_poster.png>`**:
  - Exit code: 0.
  - Output artifacts generated:
    - `images/compare.png` ($2\times 2$ grid comparison).
    - `images/before_after_360_compare.mp4` ($360^\circ$ turntable orbital comparison).

---

### 3.4. Tier 4: Headless Blender 5.2.2 LTS Verification (`scripts/verify_blender_glb.py`)

Automated verification executed in headless Blender 5.2.2 LTS via `blender-launcher.exe`:
- **Armature & Hierarchy**: Locates `ARMATURE` object and verifies all 25 core `rig:*` bones.
- **Skinned Meshes**: Locates `MESH` objects (`Alpha_Surface`, `Alpha_Joints`), verifies vertex counts, vertex groups, and active `ARMATURE` modifiers.
- **Skeletal Animation**: Verifies `Action` on Armature, $\operatorname{count}(\text{F-curves}) > 0$, and valid frame range (`frame_end >= frame_start`).
- **Headless Render**: Renders test frame to disk at $512\times 512$ PNG via camera and lighting.
- **Disk Logging & Structured JSON**: Writes all console logs and status directly to disk log file (`STATUS: PASS`), writing verification metrics to JSON.

---

## 4. Test Execution Standards & Canonical Commands

### 4.1. Python Environment Execution Standards (Windows PowerShell)

- Virtual environment path: `./venv`
- **Rule**: Always invoke pytest via the project virtual environment (`.\venv\Scripts\python.exe -m pytest`).

#### Running the Automated Test Suite:
```powershell
# Run the entire test suite with verbose output
.\venv\Scripts\python.exe -m pytest -v

# Run individual test tiers
.\venv\Scripts\python.exe -m pytest tests/test_licence_and_deps.py -v
.\venv\Scripts\python.exe -m pytest tests/test_fk_skin.py -v
.\venv\Scripts\python.exe -m pytest tests/test_retarget_stability.py -v
.\venv\Scripts\python.exe -m pytest tests/test_one_euro_filter.py -v
.\venv\Scripts\python.exe -m pytest tests/test_cli.py -v
```

### 4.2. Headless Blender 5.2.2 LTS Execution Standards

Because the Microsoft Store / MSIX package prevents direct execution of `blender.exe` (`Access is denied`), all Blender scripts must be invoked through `blender-launcher.exe`.

Because `blender-launcher.exe` detaches standard output and error from the console, scripts write logs directly to disk:

```powershell
Start-Process -FilePath "%LOCALAPPDATA%\Microsoft\WindowsApps\blender-launcher.exe" `
  -ArgumentList "--background", "--python", "scripts\verify_blender_glb.py", "--", `
    "--glb", "outputs\character_animation.glb", `
    "--log", "outputs\verify_blender.log", `
    "--json", "outputs\verify_blender.json", `
    "--render", "outputs\verify_frame.png" -Wait

# Inspect verification results
Get-Content outputs\verify_blender.log
Get-Content outputs\verify_blender.json | ConvertFrom-Json
```

---

## 5. Test Integrity & British English Compliance

1. **Zero Facade Tests**: All test suites execute genuine mathematical algorithms, AST parsers, matrix transformations, and headless render operations. Zero dummy mocks or hardcoded return assertions are permitted.
2. **Language Convention**: All docstrings, test names, log messages, error assertions, and documentation strictly adhere to **British English spellings** (*specialise*, *optimise*, *licence* (noun), *characterise*, *behaviour*, *colour*, *initialise*).

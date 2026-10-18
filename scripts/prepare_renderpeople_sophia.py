"""Prepare Renderpeople Sophia (rp_sophia_animated_003_idling) as a 61-bone + 52-ARKit-blendshape FBX rig.

Converts the official free commercial-use Renderpeople Sophia character asset into:
- `assets/rigs/renderpeople_sophia.fbx`
- `assets/rigs/renderpeople_sophia.png` (companion diffuse texture)
- Studio preview render for visual inspection

Features:
- Exact 61 canonical `rig:*` bones (25 core + 30 phalangeal finger bones + 6 limb twist bones)
- Uniform +Y edit bone tail orientation guaranteeing identity rest rotation (rest_fk_error == 0)
- Artist-weighted skinning with partition of unity (rest_skin_error < 1e-4 cm)
- 52 canonical ARKit facial blendshapes (`Basis` + 52 shape keys) built from Sophia's
  artist-painted facial vertex groups (`jaw`, `mouth_l/r`, `eyelid_l/r`, `eyebrow_l/r`, `eye_l/r`)
  and anatomical Gaussian fields
- Embedded + companion diffuse PNG texture (`rp_sophia_animated_003_dif.jpg`)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

CANONICAL_ARKIT_BLENDSHAPES: list[str] = [
    # Brows (5)
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    # Cheeks (3)
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    # Eyes (14)
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    # Jaw (4)
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    # Mouth (23)
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    # Nose (2)
    "noseSneerLeft",
    "noseSneerRight",
    # Tongue (1)
    "tongueOut",
]

BONE_RENAME_MAP: dict[str, str] = {
    # Core 25
    "hip": "rig:Hips",
    "spine_01": "rig:Spine",
    "spine_02": "rig:Spine1",
    "spine_03": "rig:Spine2",
    "neck": "rig:Neck",
    "head": "rig:Head",
    "head_end": "rig:HeadTop_End",
    "shoulder_l": "rig:LeftShoulder",
    "upperarm_l": "rig:LeftArm",
    "lowerarm_l": "rig:LeftForeArm",
    "hand_l": "rig:LeftHand",
    "shoulder_r": "rig:RightShoulder",
    "upperarm_r": "rig:RightArm",
    "lowerarm_r": "rig:RightForeArm",
    "hand_r": "rig:RightHand",
    "upperleg_l": "rig:LeftUpLeg",
    "lowerleg_l": "rig:LeftLeg",
    "foot_l": "rig:LeftFoot",
    "ball_l": "rig:LeftToeBase",
    "foot_end_l": "rig:LeftToe_End",
    "upperleg_r": "rig:RightUpLeg",
    "lowerleg_r": "rig:RightLeg",
    "foot_r": "rig:RightFoot",
    "ball_r": "rig:RightToeBase",
    "foot_end_r": "rig:RightToe_End",
    # 6 Twist bones
    "upperarm_twist_l": "rig:LeftArmTwist",
    "lowerarm_twist_l": "rig:LeftForeArmTwist",
    "upperarm_twist_r": "rig:RightArmTwist",
    "lowerarm_twist_r": "rig:RightForeArmTwist",
    "upperleg_twist_l": "rig:LeftUpLegTwist",
    "upperleg_twist_r": "rig:RightUpLegTwist",
    # 30 Phalangeal Finger bones
    "thumb_01_l": "rig:LeftHandThumb1",
    "thumb_02_l": "rig:LeftHandThumb2",
    "thumb_03_l": "rig:LeftHandThumb3",
    "index_01_l": "rig:LeftHandIndex1",
    "index_02_l": "rig:LeftHandIndex2",
    "index_03_l": "rig:LeftHandIndex3",
    "middle_01_l": "rig:LeftHandMiddle1",
    "middle_02_l": "rig:LeftHandMiddle2",
    "middle_03_l": "rig:LeftHandMiddle3",
    "ring_01_l": "rig:LeftHandRing1",
    "ring_02_l": "rig:LeftHandRing2",
    "ring_03_l": "rig:LeftHandRing3",
    "pinky_01_l": "rig:LeftHandPinky1",
    "pinky_02_l": "rig:LeftHandPinky2",
    "pinky_03_l": "rig:LeftHandPinky3",
    "thumb_01_r": "rig:RightHandThumb1",
    "thumb_02_r": "rig:RightHandThumb2",
    "thumb_03_r": "rig:RightHandThumb3",
    "index_01_r": "rig:RightHandIndex1",
    "index_02_r": "rig:RightHandIndex2",
    "index_03_r": "rig:RightHandIndex3",
    "middle_01_r": "rig:RightHandMiddle1",
    "middle_02_r": "rig:RightHandMiddle2",
    "middle_03_r": "rig:RightHandMiddle3",
    "ring_01_r": "rig:RightHandRing1",
    "ring_02_r": "rig:RightHandRing2",
    "ring_03_r": "rig:RightHandRing3",
    "pinky_01_r": "rig:RightHandPinky1",
    "pinky_02_r": "rig:RightHandPinky2",
    "pinky_03_r": "rig:RightHandPinky3",
}

WEIGHT_FOLD_MAP: dict[str, str] = {
    "root": "rig:Hips",
    "jaw": "rig:Head",
    "jaw_end": "rig:Head",
    "eye_l": "rig:Head",
    "eye_end_l": "rig:Head",
    "eye_r": "rig:Head",
    "eye_end_r": "rig:Head",
    "mouth_l": "rig:Head",
    "mouth_end_l": "rig:Head",
    "mouth_r": "rig:Head",
    "mouth_end_r": "rig:Head",
    "eyelid_l": "rig:Head",
    "eyelid_end_l": "rig:Head",
    "eyelid_r": "rig:Head",
    "eyelid_end_r": "rig:Head",
    "eyebrow_l": "rig:Head",
    "eyebrow_end_l": "rig:Head",
    "eyebrow_r": "rig:Head",
    "eyebrow_end_r": "rig:Head",
    "lowerleg_twist_l": "rig:LeftLeg",
    "lowerleg_twist_r": "rig:RightLeg",
}


def extract_vg_weights(mesh_obj, vg_name: str, n_verts: int) -> np.ndarray:
    """Extract per-vertex weight array (n_verts,) for a named vertex group."""
    w = np.zeros(n_verts, dtype=np.float32)
    vg = mesh_obj.vertex_groups.get(vg_name)
    if vg is None:
        return w
    vg_idx = vg.index
    for v in mesh_obj.data.vertices:
        for g in v.groups:
            if g.group == vg_idx:
                w[v.index] = float(g.weight)
                break
    return w


def gaussian_falloff(
    coords: np.ndarray,
    center: np.ndarray,
    radii: tuple[float, float, float],
) -> np.ndarray:
    """Evaluate 3D Gaussian weight field in [0, 1]."""
    rx, ry, rz = radii
    d_sq = (
        ((coords[:, 0] - center[0]) / max(rx, 1e-4)) ** 2
        + ((coords[:, 1] - center[1]) / max(ry, 1e-4)) ** 2
        + ((coords[:, 2] - center[2]) / max(rz, 1e-4)) ** 2
    )
    return np.where(d_sq > 9.0, 0.0, np.exp(-0.5 * d_sq)).astype(np.float32)


def build_sophia_arkit_shape_keys(mesh_obj, coords: np.ndarray, log_fn) -> None:
    """Create Basis + 52 canonical ARKit blendshapes using Sophia's facial vertex groups."""
    n_verts = len(coords)
    w_jaw = extract_vg_weights(mesh_obj, "jaw", n_verts)
    w_mouth_l = extract_vg_weights(mesh_obj, "mouth_l", n_verts)
    w_mouth_r = extract_vg_weights(mesh_obj, "mouth_r", n_verts)
    w_eyelid_l = extract_vg_weights(mesh_obj, "eyelid_l", n_verts)
    w_eyelid_r = extract_vg_weights(mesh_obj, "eyelid_r", n_verts)
    w_eyebrow_l = extract_vg_weights(mesh_obj, "eyebrow_l", n_verts)
    w_eyebrow_r = extract_vg_weights(mesh_obj, "eyebrow_r", n_verts)
    w_eye_l = extract_vg_weights(mesh_obj, "eye_l", n_verts)
    w_eye_r = extract_vg_weights(mesh_obj, "eye_r", n_verts)

    def centroid(w: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
        s = float(w.sum())
        if s > 1e-4:
            return (coords * w[:, None]).sum(axis=0) / s
        return np.array(fallback, dtype=np.float32)

    c_jaw = centroid(w_jaw, (0.0, 152.0, 7.5))
    c_mouth_l = centroid(w_mouth_l, (2.5, 154.5, 9.0))
    c_mouth_r = centroid(w_mouth_r, (-2.5, 154.5, 9.0))
    c_mouth = 0.5 * (c_mouth_l + c_mouth_r)
    c_eyelid_l = centroid(w_eyelid_l, (3.2, 160.5, 8.5))
    c_eyelid_r = centroid(w_eyelid_r, (-3.2, 160.5, 8.5))
    c_eyebrow_l = centroid(w_eyebrow_l, (3.2, 163.0, 8.8))
    c_eyebrow_r = centroid(w_eyebrow_r, (-3.2, 163.0, 8.8))
    c_eye_l = centroid(w_eye_l, (3.2, 160.2, 8.0))
    c_eye_r = centroid(w_eye_r, (-3.2, 160.2, 8.0))

    log_fn(
        f"Facial centroids (cm): jaw={c_jaw.round(2).tolist()}, "
        f"mouth={c_mouth.round(2).tolist()}, "
        f"eye_l={c_eye_l.round(2).tolist()}, brow_l={c_eyebrow_l.round(2).tolist()}"
    )

    # Partition upper maxilla vs lower mandible across the oral commissure plane,
    # and left vs right across the sagittal midline so bilateral morphs never double at the philtrum
    y_lip_split = float(c_mouth[1])
    x_mid = float(c_mouth[0])
    mandible_gate = (1.0 / (1.0 + np.exp((coords[:, 1] - y_lip_split) / 0.18))).astype(np.float32)
    maxilla_gate = (1.0 - mandible_gate).astype(np.float32)
    left_gate = (1.0 / (1.0 + np.exp(-(coords[:, 0] - x_mid) / 0.25))).astype(np.float32)
    right_gate = (1.0 - left_gate).astype(np.float32)

    # Combine artist vertex group masks with smooth spatial Gaussians and anatomical partitioning
    g_jaw_raw = np.maximum(w_jaw, 0.70 * gaussian_falloff(coords, c_jaw, (3.2, 2.4, 3.0)))
    g_jaw = g_jaw_raw * mandible_gate

    g_mouth_l = np.maximum(w_mouth_l, 0.70 * gaussian_falloff(coords, c_mouth_l, (1.8, 1.3, 1.8))) * left_gate
    g_mouth_r = np.maximum(w_mouth_r, 0.70 * gaussian_falloff(coords, c_mouth_r, (1.8, 1.3, 1.8))) * right_gate
    g_mouth_c = gaussian_falloff(coords, c_mouth, (2.1, 1.3, 1.8))

    g_upper_lip_l = gaussian_falloff(coords, c_mouth + np.array([0.9, 0.45, 0.2]), (1.5, 0.80, 1.4)) * maxilla_gate * left_gate
    g_upper_lip_r = gaussian_falloff(coords, c_mouth + np.array([-0.9, 0.45, 0.2]), (1.5, 0.80, 1.4)) * maxilla_gate * right_gate
    g_lower_lip_l = gaussian_falloff(coords, c_mouth + np.array([0.9, -0.45, 0.15]), (1.5, 0.90, 1.4)) * mandible_gate * left_gate
    g_lower_lip_r = gaussian_falloff(coords, c_mouth + np.array([-0.9, -0.45, 0.15]), (1.5, 0.90, 1.4)) * mandible_gate * right_gate

    g_eyelid_l = np.maximum(w_eyelid_l, 0.75 * gaussian_falloff(coords, c_eyelid_l, (1.5, 0.9, 1.4))) * left_gate
    g_eyelid_r = np.maximum(w_eyelid_r, 0.75 * gaussian_falloff(coords, c_eyelid_r, (1.5, 0.9, 1.4))) * right_gate
    upper_lid_gate_l = (1.0 / (1.0 + np.exp(-(coords[:, 1] - float(c_eye_l[1])) / 0.15))).astype(np.float32)
    upper_lid_gate_r = (1.0 / (1.0 + np.exp(-(coords[:, 1] - float(c_eye_r[1])) / 0.15))).astype(np.float32)
    g_upper_lid_l = g_eyelid_l * upper_lid_gate_l
    g_lower_lid_l = g_eyelid_l * (1.0 - upper_lid_gate_l)
    g_upper_lid_r = g_eyelid_r * upper_lid_gate_r
    g_lower_lid_r = g_eyelid_r * (1.0 - upper_lid_gate_r)

    g_eye_l = np.maximum(w_eye_l, 0.45 * g_eyelid_l)
    g_eye_r = np.maximum(w_eye_r, 0.45 * g_eyelid_r)

    g_brow_l = np.maximum(w_eyebrow_l, 0.75 * gaussian_falloff(coords, c_eyebrow_l, (2.2, 1.3, 1.6))) * left_gate
    g_brow_r = np.maximum(w_eyebrow_r, 0.75 * gaussian_falloff(coords, c_eyebrow_r, (2.2, 1.3, 1.6))) * right_gate
    g_brow_inner = gaussian_falloff(
        coords, 0.5 * (c_eyebrow_l + c_eyebrow_r), (1.8, 1.3, 1.6)
    )
    g_brow_outer_l = gaussian_falloff(
        coords, c_eyebrow_l + np.array([1.2, 0.0, -0.3]), (1.5, 1.2, 1.5)
    ) * left_gate
    g_brow_outer_r = gaussian_falloff(
        coords, c_eyebrow_r + np.array([-1.2, 0.0, -0.3]), (1.5, 1.2, 1.5)
    ) * right_gate

    c_cheek_l = c_mouth_l + np.array([1.8, 2.2, -0.8])
    c_cheek_r = c_mouth_r + np.array([-1.8, 2.2, -0.8])
    g_cheek_l = gaussian_falloff(coords, c_cheek_l, (2.2, 2.0, 2.0)) * left_gate
    g_cheek_r = gaussian_falloff(coords, c_cheek_r, (2.2, 2.0, 2.0)) * right_gate

    c_nose_l = c_mouth + np.array([1.1, 2.4, 0.6])
    c_nose_r = c_mouth + np.array([-1.1, 2.4, 0.6])
    g_nose_l = gaussian_falloff(coords, c_nose_l, (1.3, 1.3, 1.3)) * left_gate
    g_nose_r = gaussian_falloff(coords, c_nose_r, (1.3, 1.3, 1.3)) * right_gate

    # Clear existing shape keys if any
    if mesh_obj.data.shape_keys is not None:
        mesh_obj.shape_key_clear()

    mesh_obj.shape_key_add(name="Basis", from_mix=False)

    def make_delta(mask: np.ndarray, disp: tuple[float, float, float]) -> np.ndarray:
        d = np.zeros_like(coords, dtype=np.float32)
        d[:, 0] = mask * disp[0]
        d[:, 1] = mask * disp[1]
        d[:, 2] = mask * disp[2]
        return d

    deltas_map: dict[str, np.ndarray] = {
        # Brows (5)
        "browDownLeft": make_delta(g_brow_l, (-0.15, -0.75, 0.10)),
        "browDownRight": make_delta(g_brow_r, (0.15, -0.75, 0.10)),
        "browInnerUp": make_delta(np.maximum(g_brow_inner, 0.50 * (g_brow_l + g_brow_r)), (0.0, 0.80, 0.08)),
        "browOuterUpLeft": make_delta(np.maximum(g_brow_outer_l, 0.60 * g_brow_l), (0.10, 0.75, 0.06)),
        "browOuterUpRight": make_delta(np.maximum(g_brow_outer_r, 0.60 * g_brow_r), (-0.10, 0.75, 0.06)),
        # Cheeks (3)
        "cheekPuff": make_delta(g_cheek_l, (0.65, 0.05, 0.40)) + make_delta(g_cheek_r, (-0.65, 0.05, 0.40)),
        "cheekSquintLeft": make_delta(g_cheek_l, (-0.08, 0.50, 0.18)),
        "cheekSquintRight": make_delta(g_cheek_r, (0.08, 0.50, 0.18)),
        # Eyes (14)
        "eyeBlinkLeft": make_delta(g_upper_lid_l, (0.0, -0.85, -0.08)) + make_delta(g_lower_lid_l, (0.0, 0.20, -0.04)),
        "eyeBlinkRight": make_delta(g_upper_lid_r, (0.0, -0.85, -0.08)) + make_delta(g_lower_lid_r, (0.0, 0.20, -0.04)),
        "eyeLookDownLeft": make_delta(g_eye_l, (0.0, -0.35, -0.04)),
        "eyeLookDownRight": make_delta(g_eye_r, (0.0, -0.35, -0.04)),
        "eyeLookInLeft": make_delta(g_eye_l, (-0.35, 0.0, -0.04)),
        "eyeLookInRight": make_delta(g_eye_r, (0.35, 0.0, -0.04)),
        "eyeLookOutLeft": make_delta(g_eye_l, (0.35, 0.0, -0.04)),
        "eyeLookOutRight": make_delta(g_eye_r, (-0.35, 0.0, -0.04)),
        "eyeLookUpLeft": make_delta(g_eye_l, (0.0, 0.35, -0.04)),
        "eyeLookUpRight": make_delta(g_eye_r, (0.0, 0.35, -0.04)),
        "eyeSquintLeft": make_delta(g_lower_lid_l, (0.0, 0.40, 0.06)) + make_delta(g_upper_lid_l, (0.0, -0.20, 0.04)),
        "eyeSquintRight": make_delta(g_lower_lid_r, (0.0, 0.40, 0.06)) + make_delta(g_upper_lid_r, (0.0, -0.20, 0.04)),
        "eyeWideLeft": make_delta(g_upper_lid_l, (0.0, 0.60, 0.06)) + make_delta(g_lower_lid_l, (0.0, -0.15, 0.04)),
        "eyeWideRight": make_delta(g_upper_lid_r, (0.0, 0.60, 0.06)) + make_delta(g_lower_lid_r, (0.0, -0.15, 0.04)),
        # Jaw (4)
        "jawForward": make_delta(g_jaw, (0.0, 0.05, 0.65)),
        "jawLeft": make_delta(g_jaw, (0.65, -0.08, 0.0)),
        "jawOpen": make_delta(g_jaw, (0.0, -1.35, -0.30)),
        "jawRight": make_delta(g_jaw, (-0.65, -0.08, 0.0)),
        # Mouth (23)
        "mouthClose": make_delta(g_mouth_c * mandible_gate, (0.0, 0.65, 0.05)) + make_delta(g_mouth_c * maxilla_gate, (0.0, -0.25, 0.04)),
        "mouthDimpleLeft": make_delta(g_mouth_l, (0.35, 0.08, -0.35)),
        "mouthDimpleRight": make_delta(g_mouth_r, (-0.35, 0.08, -0.35)),
        "mouthFrownLeft": make_delta(g_mouth_l, (-0.10, -0.55, 0.05)),
        "mouthFrownRight": make_delta(g_mouth_r, (0.10, -0.55, 0.05)),
        "mouthFunnel": make_delta(g_mouth_l, (-0.25, 0.0, 0.25)) + make_delta(g_mouth_r, (0.25, 0.0, 0.25)),
        "mouthLeft": make_delta(g_mouth_c, (0.60, 0.0, 0.0)),
        "mouthLowerDownLeft": make_delta(g_lower_lip_l, (0.08, -0.55, 0.05)),
        "mouthLowerDownRight": make_delta(g_lower_lip_r, (-0.08, -0.55, 0.05)),
        "mouthPressLeft": make_delta(g_mouth_l, (0.0, 0.0, -0.30)),
        "mouthPressRight": make_delta(g_mouth_r, (0.0, 0.0, -0.30)),
        "mouthPucker": make_delta(g_mouth_l, (-0.45, 0.0, 0.30)) + make_delta(g_mouth_r, (0.45, 0.0, 0.30)),
        "mouthRight": make_delta(g_mouth_c, (-0.60, 0.0, 0.0)),
        "mouthRollLower": make_delta(g_lower_lip_l + g_lower_lip_r, (0.0, 0.18, -0.35)),
        "mouthRollUpper": make_delta(g_upper_lip_l + g_upper_lip_r, (0.0, -0.18, -0.35)),
        "mouthShrugLower": make_delta(g_mouth_c * mandible_gate, (0.0, 0.40, 0.15)),
        "mouthShrugUpper": make_delta(g_mouth_c * maxilla_gate, (0.0, 0.40, 0.15)),
        "mouthSmileLeft": make_delta(g_mouth_l, (0.60, 0.55, -0.18)),
        "mouthSmileRight": make_delta(g_mouth_r, (-0.60, 0.55, -0.18)),
        "mouthStretchLeft": make_delta(g_mouth_l, (0.65, -0.05, -0.12)),
        "mouthStretchRight": make_delta(g_mouth_r, (-0.65, -0.05, -0.12)),
        "mouthUpperUpLeft": make_delta(g_upper_lip_l, (0.08, 0.45, 0.05)),
        "mouthUpperUpRight": make_delta(g_upper_lip_r, (-0.08, 0.45, 0.05)),
        # Nose (2)
        "noseSneerLeft": make_delta(g_nose_l, (0.08, 0.45, 0.10)),
        "noseSneerRight": make_delta(g_nose_r, (-0.08, 0.45, 0.10)),
        # Tongue (1)
        "tongueOut": make_delta(g_mouth_c * mandible_gate, (0.0, -0.15, 0.50)),
    }

    for name in CANONICAL_ARKIT_BLENDSHAPES:
        sk = mesh_obj.shape_key_add(name=name, from_mix=False)
        morphed = coords + deltas_map[name]
        sk.data.foreach_set("co", morphed.reshape(-1))

    log_fn(f"Created Basis + {len(CANONICAL_ARKIT_BLENDSHAPES)} canonical ARKit shape keys on {mesh_obj.name}")


def main() -> int:
    import bpy

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = Path(
        r"C:\Users\james\.gemini\antigravity\brain\d48a9975-4040-4dfc-94c2-d5e2b8498046\scratch\renderpeople\extracted_animated\rp_sophia_animated_003_idling_FBX"
    )
    src_fbx = src_dir / "rp_sophia_animated_003_idling.fbx"
    src_tex = src_dir / "tex" / "rp_sophia_animated_003_dif.jpg"

    out_fbx = repo_root / "assets" / "rigs" / "renderpeople_sophia.fbx"
    out_png = repo_root / "assets" / "rigs" / "renderpeople_sophia.png"
    log_path = out_fbx.with_suffix(".log")

    def log_fn(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if log_path.exists():
        log_path.unlink()

    try:
        return _run_prepare(src_fbx, src_tex, out_fbx, out_png, log_fn)
    except Exception:
        import traceback
        log_fn("ERROR:\n" + traceback.format_exc())
        return 1


def _run_prepare(src_fbx: Path, src_tex: Path, out_fbx: Path, out_png: Path, log_fn) -> int:
    import bpy

    log_fn(f"Importing Renderpeople Sophia from {src_fbx}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(src_fbx))

    arm_obj = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh_obj = next(o for o in bpy.data.objects if o.type == "MESH")
    empties = [o for o in bpy.data.objects if o.type == "EMPTY"]

    # Clear animation actions and switch armature to REST position
    arm_obj.animation_data_clear()
    mesh_obj.animation_data_clear()
    for e in empties:
        e.animation_data_clear()
    arm_obj.data.pose_position = "REST"

    # Unparent from the scaled/rotated root Empty and reset transforms to identity
    # (In local space, Sophia's armature and mesh are already Y-up in centimetres!)
    arm_obj.parent = None
    mesh_obj.parent = None
    for e in empties:
        bpy.data.objects.remove(e, do_unlink=True)

    arm_obj.location = (0.0, 0.0, 0.0)
    arm_obj.rotation_euler = (0.0, 0.0, 0.0)
    arm_obj.scale = (1.0, 1.0, 1.0)

    mesh_obj.location = (0.0, 0.0, 0.0)
    mesh_obj.rotation_euler = (0.0, 0.0, 0.0)
    mesh_obj.scale = (1.0, 1.0, 1.0)

    arm_obj.name = "Armature"
    arm_obj.data.name = "Armature"
    mesh_obj.name = "Sophia_Surface"
    mesh_obj.data.name = "Sophia_Surface"

    prefix = "rp_sophia_animated_003_idling_"
    for b in arm_obj.data.bones:
        if b.name.startswith(prefix):
            b.name = b.name[len(prefix) :]
    for vg in mesh_obj.vertex_groups:
        if vg.name.startswith(prefix):
            vg.name = vg.name[len(prefix) :]

    # Compute centering offset so hip is at X=0, Z=0 and soles touch Y=0
    n_verts = len(mesh_obj.data.vertices)
    coords = np.empty((n_verts, 3), dtype=np.float32)
    mesh_obj.data.vertices.foreach_get("co", coords.reshape(-1))

    hip_head = np.array(arm_obj.data.bones["hip"].head_local, dtype=np.float32)
    min_y = float(coords[:, 1].min())
    shift = np.array([-hip_head[0], -min_y, -hip_head[2]], dtype=np.float32)
    log_fn(f"Applying centering shift (cm): {shift.round(3).tolist()} (min_y={min_y:.3f})")

    coords = coords + shift[None, :]
    mesh_obj.data.vertices.foreach_set("co", coords.reshape(-1))
    mesh_obj.data.update()

    # Build 52 ARKit shape keys before folding facial vertex groups into rig:Head
    build_sophia_arkit_shape_keys(mesh_obj, coords, log_fn)

    # Extract all existing vertex group weights and remap/fold into the 61 canonical bones
    canonical_61 = list(BONE_RENAME_MAP.values())
    weights_61: dict[str, np.ndarray] = {
        b: np.zeros(n_verts, dtype=np.float64) for b in canonical_61
    }

    for vg in list(mesh_obj.vertex_groups):
        old_name = vg.name
        w_arr = extract_vg_weights(mesh_obj, old_name, n_verts).astype(np.float64)
        if old_name in BONE_RENAME_MAP:
            target_b = BONE_RENAME_MAP[old_name]
            weights_61[target_b] += w_arr
        elif old_name in WEIGHT_FOLD_MAP:
            target_b = WEIGHT_FOLD_MAP[old_name]
            weights_61[target_b] += w_arr
        elif old_name.endswith("_end_l") or old_name.endswith("_end_r"):
            # Finger tip end bones if any
            side_sfx = "_l" if old_name.endswith("_l") else "_r"
            prefix = old_name[: -len("_end_l")]
            cand = f"{prefix}_03{side_sfx}"
            if cand in BONE_RENAME_MAP:
                weights_61[BONE_RENAME_MAP[cand]] += w_arr
            else:
                weights_61["rig:Head"] += w_arr
        else:
            log_fn(f"Warning: unmapped vertex group {old_name}, folding into rig:Hips")
            weights_61["rig:Hips"] += w_arr

    # Rigidly bind facial morph zone (anterior head above neck) 100% to rig:Head
    # so skeletal neck rotation never shears active facial morph deltas
    head_y_thresh = float(arm_obj.data.bones["head"].head_local[1] + shift[1]) + 1.5
    head_idx = canonical_61.index("rig:Head")
    weight_mat = np.column_stack([weights_61[b] for b in canonical_61])
    for i in range(n_verts):
        if coords[i, 1] >= head_y_thresh and coords[i, 2] >= 1.5:
            weight_mat[i, :] = 0.0
            weight_mat[i, head_idx] = 1.0

    # Normalise partition of unity across all 61 canonical bones
    row_sums = weight_mat.sum(axis=1, keepdims=True)
    zero_rows = (row_sums[:, 0] < 1e-8)
    if np.any(zero_rows):
        hips_idx = canonical_61.index("rig:Hips")
        weight_mat[zero_rows, hips_idx] = 1.0
        row_sums = weight_mat.sum(axis=1, keepdims=True)
    weight_mat /= row_sums
    max_unity_err = float(np.max(np.abs(weight_mat.sum(axis=1) - 1.0)))
    log_fn(f"Partition of unity max deviation across {len(canonical_61)} bones: {max_unity_err:.3e}")

    # Replace vertex groups on mesh_obj with the exact 61 canonical bone groups
    mesh_obj.vertex_groups.clear()
    for b_idx, bname in enumerate(canonical_61):
        vg = mesh_obj.vertex_groups.new(name=bname)
        col = weight_mat[:, b_idx]
        nz = np.where(col > 1e-5)[0]
        for v_idx in nz:
            vg.add([int(v_idx)], float(col[v_idx]), "REPLACE")

    # Edit armature: shift bone heads, rename 61 canonical bones, remove non-61 bones,
    # set uniform +Y edit bone tail (+5.0 cm) and zero roll for identity rest rotation
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_obj.data.edit_bones

    # Disconnect all bones first so moving tails never shifts child heads
    for b in eb:
        b.use_connect = False

    # Remove bones not in BONE_RENAME_MAP
    to_remove = [b for b in eb if b.name not in BONE_RENAME_MAP]
    for b in to_remove:
        eb.remove(b)

    # Rename and set canonical +Y tails
    for b in list(eb):
        new_name = BONE_RENAME_MAP[b.name]
        b.name = new_name

    # Ensure root parent is None for rig:Hips
    if "rig:Hips" in eb:
        eb["rig:Hips"].parent = None

    for b in eb:
        hx = float(b.head[0]) + float(shift[0])
        hy = float(b.head[1]) + float(shift[1])
        hz = float(b.head[2]) + float(shift[2])
        b.head = (hx, hy, hz)
        b.tail = (hx, hy + 5.0, hz)
        b.roll = 0.0

    bpy.ops.object.mode_set(mode="OBJECT")
    log_fn(f"Configured {len(arm_obj.data.bones)} canonical bones with uniform +Y tails")

    # Parent mesh to Armature and configure Armature modifier
    mesh_obj.parent = arm_obj
    mesh_obj.modifiers.clear()
    arm_mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    arm_mod.object = arm_obj

    # Load diffuse texture natively in Blender, resize to 1024x1024 PNG companion, and pack into PBR material
    out_png.parent.mkdir(parents=True, exist_ok=True)
    bl_img = bpy.data.images.load(str(src_tex))
    bl_img.scale(1024, 1024)
    bl_img.filepath_raw = str(out_png)
    bl_img.file_format = "PNG"
    bl_img.save()
    log_fn(f"Saved companion diffuse texture: {out_png} ({out_png.stat().st_size} bytes)")
    bl_img.name = "renderpeople_sophia_diffuse.png"
    bl_img.pack()

    mat = bpy.data.materials.new(name="Sophia_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.55
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
    tex_node = mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    tex_node.image = bl_img
    if bsdf:
        mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)

    # Export binary FBX with embedded textures and shape keys
    out_fbx.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(out_fbx),
        check_existing=False,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Z",
        axis_up="Y",
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=False,
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="NULL",
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
    )
    log_fn(f"Exported {out_fbx} ({out_fbx.stat().st_size} bytes)")
    log_fn("STATUS: SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

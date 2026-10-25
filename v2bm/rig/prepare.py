"""General-purpose character rig preparation for VideoIntoBlenderModel (`v2bm prepare-rig`).

Converts third-party humanoid FBX character assets (such as Renderpeople, Mixamo,
or custom rigged characters) into a canonical 61-bone + 52-ARKit-blendshape FBX rig
ready for `v2bm run`:

- Strips character-specific bone and vertex-group prefixes automatically
- Unparents from scaled/rotated root empties and centres the character (pelvis at X=0, Z=0; soles at Y=0)
- Synthesises `Basis` + 52 canonical ARKit facial blendshapes from artist-painted facial
  vertex groups (`jaw`, `mouth_l/r`, `eyelid_l/r`, `eyebrow_l/r`, `eye_l/r`) when present,
  or from proportional anatomical head geometry when facial vertex groups are absent
- Maps and folds skinning weights into the 61 canonical `rig:*` bones with partition of unity
- Sets uniform +Y edit bone tails (`+5.0 cm`, zero roll) guaranteeing zero rest FK error
- Auto-discovers, packs, and saves a companion diffuse PNG texture (`<output_stem>.png`)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

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
    "hips": "rig:Hips",
    "pelvis": "rig:Hips",
    "spine_01": "rig:Spine",
    "spine": "rig:Spine",
    "spine_02": "rig:Spine1",
    "spine1": "rig:Spine1",
    "spine_03": "rig:Spine2",
    "spine2": "rig:Spine2",
    "neck": "rig:Neck",
    "head": "rig:Head",
    "head_end": "rig:HeadTop_End",
    "headtop_end": "rig:HeadTop_End",
    "shoulder_l": "rig:LeftShoulder",
    "leftshoulder": "rig:LeftShoulder",
    "upperarm_l": "rig:LeftArm",
    "leftarm": "rig:LeftArm",
    "lowerarm_l": "rig:LeftForeArm",
    "leftforearm": "rig:LeftForeArm",
    "hand_l": "rig:LeftHand",
    "lefthand": "rig:LeftHand",
    "shoulder_r": "rig:RightShoulder",
    "rightshoulder": "rig:RightShoulder",
    "upperarm_r": "rig:RightArm",
    "rightarm": "rig:RightArm",
    "lowerarm_r": "rig:RightForeArm",
    "rightforearm": "rig:RightForeArm",
    "hand_r": "rig:RightHand",
    "righthand": "rig:RightHand",
    "upperleg_l": "rig:LeftUpLeg",
    "leftupleg": "rig:LeftUpLeg",
    "lowerleg_l": "rig:LeftLeg",
    "leftleg": "rig:LeftLeg",
    "foot_l": "rig:LeftFoot",
    "leftfoot": "rig:LeftFoot",
    "ball_l": "rig:LeftToeBase",
    "lefttoebase": "rig:LeftToeBase",
    "foot_end_l": "rig:LeftToe_End",
    "lefttoe_end": "rig:LeftToe_End",
    "upperleg_r": "rig:RightUpLeg",
    "rightupleg": "rig:RightUpLeg",
    "lowerleg_r": "rig:RightLeg",
    "rightleg": "rig:RightLeg",
    "foot_r": "rig:RightFoot",
    "rightfoot": "rig:RightFoot",
    "ball_r": "rig:RightToeBase",
    "righttoebase": "rig:RightToeBase",
    "foot_end_r": "rig:RightToe_End",
    "righttoe_end": "rig:RightToe_End",
    # 6 Twist bones
    "upperarm_twist_l": "rig:LeftArmTwist",
    "leftarmtwist": "rig:LeftArmTwist",
    "lowerarm_twist_l": "rig:LeftForeArmTwist",
    "leftforearmtwist": "rig:LeftForeArmTwist",
    "upperarm_twist_r": "rig:RightArmTwist",
    "rightarmtwist": "rig:RightArmTwist",
    "lowerarm_twist_r": "rig:RightForeArmTwist",
    "rightforearmtwist": "rig:RightForeArmTwist",
    "upperleg_twist_l": "rig:LeftUpLegTwist",
    "leftuplegtwist": "rig:LeftUpLegTwist",
    "upperleg_twist_r": "rig:RightUpLegTwist",
    "rightuplegtwist": "rig:RightUpLegTwist",
    # 30 Phalangeal Finger bones (Unreal / Renderpeople naming)
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
    # 30 Phalangeal Finger bones (Mixamo / Make-It-Animatable naming)
    "lefthandthumb1": "rig:LeftHandThumb1",
    "lefthandthumb2": "rig:LeftHandThumb2",
    "lefthandthumb3": "rig:LeftHandThumb3",
    "lefthandindex1": "rig:LeftHandIndex1",
    "lefthandindex2": "rig:LeftHandIndex2",
    "lefthandindex3": "rig:LeftHandIndex3",
    "lefthandmiddle1": "rig:LeftHandMiddle1",
    "lefthandmiddle2": "rig:LeftHandMiddle2",
    "lefthandmiddle3": "rig:LeftHandMiddle3",
    "lefthandring1": "rig:LeftHandRing1",
    "lefthandring2": "rig:LeftHandRing2",
    "lefthandring3": "rig:LeftHandRing3",
    "lefthandpinky1": "rig:LeftHandPinky1",
    "lefthandpinky2": "rig:LeftHandPinky2",
    "lefthandpinky3": "rig:LeftHandPinky3",
    "righthandthumb1": "rig:RightHandThumb1",
    "righthandthumb2": "rig:RightHandThumb2",
    "righthandthumb3": "rig:RightHandThumb3",
    "righthandindex1": "rig:RightHandIndex1",
    "righthandindex2": "rig:RightHandIndex2",
    "righthandindex3": "rig:RightHandIndex3",
    "righthandmiddle1": "rig:RightHandMiddle1",
    "righthandmiddle2": "rig:RightHandMiddle2",
    "righthandmiddle3": "rig:RightHandMiddle3",
    "righthandring1": "rig:RightHandRing1",
    "righthandring2": "rig:RightHandRing2",
    "righthandring3": "rig:RightHandRing3",
    "righthandpinky1": "rig:RightHandPinky1",
    "righthandpinky2": "rig:RightHandPinky2",
    "righthandpinky3": "rig:RightHandPinky3",
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
    # Mixamo / Make-It-Animatable leaf end bones
    "lefthandthumb4": "rig:LeftHandThumb3",
    "lefthandindex4": "rig:LeftHandIndex3",
    "lefthandmiddle4": "rig:LeftHandMiddle3",
    "lefthandring4": "rig:LeftHandRing3",
    "lefthandpinky4": "rig:LeftHandPinky3",
    "righthandthumb4": "rig:RightHandThumb3",
    "righthandindex4": "rig:RightHandIndex3",
    "righthandmiddle4": "rig:RightHandMiddle3",
    "righthandring4": "rig:RightHandRing3",
    "righthandpinky4": "rig:RightHandPinky3",
}


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


def _derive_anatomical_face_fallbacks(
    coords: np.ndarray,
    head_pos: np.ndarray | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Derive anatomical facial landmark centroids from head geometry when vertex groups are absent."""
    if head_pos is not None:
        hx, hy, hz = float(head_pos[0]), float(head_pos[1]), float(head_pos[2])
    else:
        max_y = float(coords[:, 1].max())
        hx, hy, hz = 0.0, max_y - 22.0, 0.0

    head_mask = coords[:, 1] >= hy
    head_verts = coords[head_mask] if np.any(head_mask) else coords
    max_y = float(head_verts[:, 1].max())
    head_h = max(max_y - hy, 12.0)
    z_front = float(np.percentile(head_verts[:, 2], 95)) if len(head_verts) > 0 else hz + 9.0

    return {
        "jaw": (hx, hy + 0.12 * head_h, z_front - 1.5),
        "mouth_l": (hx + 0.11 * head_h, hy + 0.24 * head_h, z_front - 0.5),
        "mouth_r": (hx - 0.11 * head_h, hy + 0.24 * head_h, z_front - 0.5),
        "eyelid_l": (hx + 0.14 * head_h, hy + 0.52 * head_h, z_front - 0.8),
        "eyelid_r": (hx - 0.14 * head_h, hy + 0.52 * head_h, z_front - 0.8),
        "eyebrow_l": (hx + 0.14 * head_h, hy + 0.63 * head_h, z_front - 0.5),
        "eyebrow_r": (hx - 0.14 * head_h, hy + 0.63 * head_h, z_front - 0.5),
        "eye_l": (hx + 0.14 * head_h, hy + 0.50 * head_h, z_front - 1.2),
        "eye_r": (hx - 0.14 * head_h, hy + 0.50 * head_h, z_front - 1.2),
    }


def compute_arkit_morph_deltas(
    coords: np.ndarray,
    *,
    head_pos: np.ndarray | None = None,
    vg_weights: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute 52 canonical ARKit blendshape displacement deltas `(V, 3)` in FBX cm.

    Uses artist-painted facial vertex groups (`jaw`, `mouth_l/r`, `eyelid_l/r`,
    `eyebrow_l/r`, `eye_l/r`) when provided, and automatically falls back to
    proportional anatomical head geometry derived from `coords` and `head_pos`.
    """
    coords_f32 = np.asarray(coords, dtype=np.float32)
    n_verts = len(coords_f32)
    vg = vg_weights or {}
    zeros = np.zeros(n_verts, dtype=np.float32)

    w_jaw = vg.get("jaw", zeros)
    w_mouth_l = vg.get("mouth_l", zeros)
    w_mouth_r = vg.get("mouth_r", zeros)
    w_eyelid_l = vg.get("eyelid_l", zeros)
    w_eyelid_r = vg.get("eyelid_r", zeros)
    w_eyebrow_l = vg.get("eyebrow_l", zeros)
    w_eyebrow_r = vg.get("eyebrow_r", zeros)
    w_eye_l = vg.get("eye_l", zeros)
    w_eye_r = vg.get("eye_r", zeros)

    fallbacks = _derive_anatomical_face_fallbacks(coords_f32, head_pos=head_pos)

    def centroid(w: np.ndarray, key: str) -> np.ndarray:
        s = float(w.sum())
        if s > 1e-4:
            return (coords_f32 * w[:, None]).sum(axis=0) / s
        return np.array(fallbacks[key], dtype=np.float32)

    c_jaw = centroid(w_jaw, "jaw")

    def resolve_bilateral(
        w_l: np.ndarray, w_r: np.ndarray, key_l: str, key_r: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c_l = centroid(w_l, key_l)
        c_r = centroid(w_r, key_r)
        if c_l[0] < c_r[0]:
            w_l, w_r = w_r, w_l
            c_l, c_r = c_r, c_l
        return w_l, w_r, c_l, c_r

    w_mouth_l, w_mouth_r, c_mouth_l, c_mouth_r = resolve_bilateral(
        w_mouth_l, w_mouth_r, "mouth_l", "mouth_r"
    )
    c_mouth = 0.5 * (c_mouth_l + c_mouth_r)
    w_eyelid_l, w_eyelid_r, c_eyelid_l, c_eyelid_r = resolve_bilateral(
        w_eyelid_l, w_eyelid_r, "eyelid_l", "eyelid_r"
    )
    w_eyebrow_l, w_eyebrow_r, c_eyebrow_l, c_eyebrow_r = resolve_bilateral(
        w_eyebrow_l, w_eyebrow_r, "eyebrow_l", "eyebrow_r"
    )
    w_eye_l, w_eye_r, c_eye_l, c_eye_r = resolve_bilateral(
        w_eye_l, w_eye_r, "eye_l", "eye_r"
    )

    centroids = {
        "jaw": c_jaw,
        "mouth": c_mouth,
        "eye_l": c_eye_l,
        "eye_r": c_eye_r,
        "brow_l": c_eyebrow_l,
        "brow_r": c_eyebrow_r,
    }

    # Partition upper maxilla vs lower mandible across the oral commissure plane,
    # and left vs right across the sagittal midline so bilateral morphs never double at the philtrum
    y_lip_split = float(c_mouth[1])
    x_mid = float(c_mouth[0])
    mandible_gate = (1.0 / (1.0 + np.exp((coords_f32[:, 1] - y_lip_split) / 0.18))).astype(np.float32)
    maxilla_gate = (1.0 - mandible_gate).astype(np.float32)
    left_gate = (1.0 / (1.0 + np.exp(-(coords_f32[:, 0] - x_mid) / 0.25))).astype(np.float32)
    right_gate = (1.0 - left_gate).astype(np.float32)

    # Combine artist vertex group masks with smooth spatial Gaussians and anatomical partitioning
    g_jaw_raw = np.maximum(w_jaw, 0.70 * gaussian_falloff(coords_f32, c_jaw, (3.2, 2.4, 3.0)))
    g_jaw = g_jaw_raw * mandible_gate

    g_mouth_l = np.maximum(w_mouth_l, 0.70 * gaussian_falloff(coords_f32, c_mouth_l, (1.8, 1.3, 1.8))) * left_gate
    g_mouth_r = np.maximum(w_mouth_r, 0.70 * gaussian_falloff(coords_f32, c_mouth_r, (1.8, 1.3, 1.8))) * right_gate
    g_mouth_c = gaussian_falloff(coords_f32, c_mouth, (2.1, 1.3, 1.8))

    g_upper_lip_l = gaussian_falloff(coords_f32, c_mouth + np.array([0.9, 0.45, 0.2]), (1.5, 0.80, 1.4)) * maxilla_gate * left_gate
    g_upper_lip_r = gaussian_falloff(coords_f32, c_mouth + np.array([-0.9, 0.45, 0.2]), (1.5, 0.80, 1.4)) * maxilla_gate * right_gate
    g_lower_lip_l = gaussian_falloff(coords_f32, c_mouth + np.array([0.9, -0.45, 0.15]), (1.5, 0.90, 1.4)) * mandible_gate * left_gate
    g_lower_lip_r = gaussian_falloff(coords_f32, c_mouth + np.array([-0.9, -0.45, 0.15]), (1.5, 0.90, 1.4)) * mandible_gate * right_gate

    g_eyelid_l = np.maximum(w_eyelid_l, 0.75 * gaussian_falloff(coords_f32, c_eyelid_l, (1.5, 0.9, 1.4))) * left_gate
    g_eyelid_r = np.maximum(w_eyelid_r, 0.75 * gaussian_falloff(coords_f32, c_eyelid_r, (1.5, 0.9, 1.4))) * right_gate
    upper_lid_gate_l = (1.0 / (1.0 + np.exp(-(coords_f32[:, 1] - float(c_eye_l[1])) / 0.15))).astype(np.float32)
    upper_lid_gate_r = (1.0 / (1.0 + np.exp(-(coords_f32[:, 1] - float(c_eye_r[1])) / 0.15))).astype(np.float32)
    g_upper_lid_l = g_eyelid_l * upper_lid_gate_l
    g_lower_lid_l = g_eyelid_l * (1.0 - upper_lid_gate_l)
    g_upper_lid_r = g_eyelid_r * upper_lid_gate_r
    g_lower_lid_r = g_eyelid_r * (1.0 - upper_lid_gate_r)

    g_eye_l = np.maximum(w_eye_l, 0.45 * g_eyelid_l)
    g_eye_r = np.maximum(w_eye_r, 0.45 * g_eyelid_r)

    g_brow_l = np.maximum(w_eyebrow_l, 0.75 * gaussian_falloff(coords_f32, c_eyebrow_l, (2.2, 1.3, 1.6))) * left_gate
    g_brow_r = np.maximum(w_eyebrow_r, 0.75 * gaussian_falloff(coords_f32, c_eyebrow_r, (2.2, 1.3, 1.6))) * right_gate
    g_brow_inner = gaussian_falloff(
        coords_f32, 0.5 * (c_eyebrow_l + c_eyebrow_r), (1.8, 1.3, 1.6)
    )
    g_brow_outer_l = gaussian_falloff(
        coords_f32, c_eyebrow_l + np.array([1.2, 0.0, -0.3]), (1.5, 1.2, 1.5)
    ) * left_gate
    g_brow_outer_r = gaussian_falloff(
        coords_f32, c_eyebrow_r + np.array([-1.2, 0.0, -0.3]), (1.5, 1.2, 1.5)
    ) * right_gate

    c_cheek_l = c_mouth_l + np.array([1.8, 2.2, -0.8])
    c_cheek_r = c_mouth_r + np.array([-1.8, 2.2, -0.8])
    g_cheek_l = gaussian_falloff(coords_f32, c_cheek_l, (2.2, 2.0, 2.0)) * left_gate
    g_cheek_r = gaussian_falloff(coords_f32, c_cheek_r, (2.2, 2.0, 2.0)) * right_gate

    c_nose_l = c_mouth + np.array([1.1, 2.4, 0.6])
    c_nose_r = c_mouth + np.array([-1.1, 2.4, 0.6])
    g_nose_l = gaussian_falloff(coords_f32, c_nose_l, (1.3, 1.3, 1.3)) * left_gate
    g_nose_r = gaussian_falloff(coords_f32, c_nose_r, (1.3, 1.3, 1.3)) * right_gate

    def make_delta(mask: np.ndarray, disp: tuple[float, float, float]) -> np.ndarray:
        d = np.zeros_like(coords_f32, dtype=np.float32)
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
    return deltas_map, centroids


def detect_common_bone_prefix(bone_names: list[str]) -> str:
    """Auto-detect a common naming prefix across armature bones (e.g. 'rp_nathan_animated_003_walking_')."""
    lower_set = {b.lower() for b in bone_names}
    for bname in bone_names:
        low = bname.lower()
        for hip_token in ("hip", "hips", "pelvis"):
            if low.endswith(hip_token) and len(bname) > len(hip_token):
                candidate = bname[: len(bname) - len(hip_token)]
                cand_low = candidate.lower()
                if any((cand_low + h) in lower_set for h in ("head", "neck", "spine", "spine_01")):
                    return candidate
    return ""


def discover_diffuse_texture(src_fbx: Path) -> Path | None:
    """Auto-discover the primary diffuse/albedo texture accompanying an FBX file."""
    search_dirs = [
        src_fbx.parent / "tex",
        src_fbx.parent / "textures",
        src_fbx.parent / f"{src_fbx.stem}.fbm",
        src_fbx.parent,
        src_fbx.parent.parent / "tex",
    ]
    diffuse_hints = ("_dif", "diffuse", "albedo", "basecolor", "base_color", "_col", "color")
    other_hints = ("norm", "_nrm", "spec", "rough", "metal", "gloss", "bump", "_ao", "opacity", "alpha")

    best_candidates: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            low = p.stem.lower()
            if any(h in low for h in diffuse_hints):
                best_candidates.append((0, p))
            elif not any(h in low for h in other_hints):
                best_candidates.append((1, p))

    if not best_candidates:
        return None
    best_candidates.sort(key=lambda item: item[0])
    return best_candidates[0][1]


def extract_vg_weights(mesh_obj, vg_name: str, n_verts: int) -> np.ndarray:
    """Extract per-vertex weight array (n_verts,) for a named Blender vertex group."""
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


def build_arkit_shape_keys_in_blender(
    mesh_obj,
    coords: np.ndarray,
    head_pos: np.ndarray | None,
    log_fn: Callable[[str], None],
) -> None:
    """Create Basis + 52 canonical ARKit blendshapes on a Blender mesh object."""
    n_verts = len(coords)
    vg_names = (
        "jaw",
        "mouth_l",
        "mouth_r",
        "eyelid_l",
        "eyelid_r",
        "eyebrow_l",
        "eyebrow_r",
        "eye_l",
        "eye_r",
    )
    vg_weights = {name: extract_vg_weights(mesh_obj, name, n_verts) for name in vg_names}
    deltas_map, centroids = compute_arkit_morph_deltas(
        coords, head_pos=head_pos, vg_weights=vg_weights
    )

    log_fn(
        f"Facial centroids (cm): jaw={centroids['jaw'].round(2).tolist()}, "
        f"mouth={centroids['mouth'].round(2).tolist()}, "
        f"eye_l={centroids['eye_l'].round(2).tolist()}, brow_l={centroids['brow_l'].round(2).tolist()}"
    )

    if mesh_obj.data.shape_keys is not None:
        mesh_obj.shape_key_clear()

    mesh_obj.shape_key_add(name="Basis", from_mix=False)
    for name in CANONICAL_ARKIT_BLENDSHAPES:
        sk = mesh_obj.shape_key_add(name=name, from_mix=False)
        morphed = coords + deltas_map[name]
        sk.data.foreach_set("co", morphed.reshape(-1))

    log_fn(
        f"Created Basis + {len(CANONICAL_ARKIT_BLENDSHAPES)} canonical ARKit shape keys on {mesh_obj.name}"
    )


def prune_topological_webbing_in_blender(
    mesh_obj,
    log_fn: Callable[[str], None],
) -> int:
    """Detect and prune topological bridging webbing and kinematic branch cross-connections

    (stretched polygons connecting disparate limbs or cross-body contact surfaces generated by
    2D-to-3D neural reconstruction when an unbent rest pose is applied).
    """
    import bmesh

    verts_co = np.array([v.co for v in mesh_obj.data.vertices], dtype=np.float32)
    if len(verts_co) == 0 or len(mesh_obj.data.edges) == 0 or len(mesh_obj.data.polygons) == 0:
        return 0

    bbox_dim = verts_co.max(axis=0) - verts_co.min(axis=0)
    char_dim = float(max(bbox_dim))

    edge_lens = [
        np.linalg.norm(verts_co[e.vertices[0]] - verts_co[e.vertices[1]])
        for e in mesh_obj.data.edges
    ]
    med_edge = float(np.median(edge_lens))

    # 1. Scale-invariant stretch threshold:
    # A single polygon edge must exceed 16% of total character dimension AND 18x median edge length
    stretch_thresh = max(0.16 * char_dim, 18.0 * med_edge)
    kinematic_bridge_min_edge = max(0.10 * char_dim, 10.0 * med_edge)

    # 2. Kinematic limb classification from existing vertex groups
    def classify_limb(bname: str) -> str:
        low = bname.lower()
        if any(
            k in low
            for k in (
                "lefthand",
                "leftarm",
                "leftforearm",
                "leftshoulder",
                "leftfinger",
                "leftthumb",
                "leftindex",
                "leftmiddle",
                "leftring",
                "leftpinky",
            )
        ):
            return "L_ARM"
        if any(
            k in low
            for k in (
                "righthand",
                "rightarm",
                "rightforearm",
                "rightshoulder",
                "rightfinger",
                "rightthumb",
                "rightindex",
                "rightmiddle",
                "rightring",
                "rightpinky",
            )
        ):
            return "R_ARM"
        if "head" in low:
            return "HEAD"
        if any(k in low for k in ("leftleg", "leftfoot", "lefttoe")):
            return "L_LEG"
        if any(k in low for k in ("rightleg", "rightfoot", "righttoe")):
            return "R_LEG"
        return "OTHER"

    vg_limbs = {vg.index: classify_limb(vg.name) for vg in mesh_obj.vertex_groups}

    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)

    # 0. Weld coincident UV-seam split vertices (scale-invariant: 0.005% of character dimension)
    # while preserving per-loop UV coordinates and per-vertex deform weights so surface topology is manifold.
    n_verts_before_weld = len(bm.verts)
    weld_dist = max(1e-4, 0.00005 * char_dim)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld_dist)
    n_welded = n_verts_before_weld - len(bm.verts)
    if n_welded > 0:
        log_fn(f"Welded {n_welded} coincident UV-seam split vertices (preserving loop UVs)")
        bm.to_mesh(mesh_obj.data)
        mesh_obj.data.update()
        verts_co = np.array([v.co for v in mesh_obj.data.vertices], dtype=np.float32)

    vert_branches: list[set[str]] = []
    for v in mesh_obj.data.vertices:
        branches: set[str] = set()
        for g in v.groups:
            if g.weight > 0.15:
                b = vg_limbs.get(g.group, "OTHER")
                if b != "OTHER":
                    branches.add(b)
        vert_branches.append(branches)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    faces_to_prune = []
    reasons = {"stretch": 0, "kinematic": 0, "sliver_islands": 0}

    for f in bm.faces:
        # Check geometric stretch
        edge_max = max(e.calc_length() for e in f.edges)
        if edge_max > stretch_thresh:
            faces_to_prune.append(f)
            reasons["stretch"] += 1
            continue

        # Check kinematic limb incompatibility (only for stretched bridging faces > kinematic_bridge_min_edge)
        if edge_max > kinematic_bridge_min_edge:
            f_vert_indices = [v.index for v in f.verts]
            b_set = set.union(*(vert_branches[vi] for vi in f_vert_indices)) if f_vert_indices else set()
            if (
                ("L_ARM" in b_set and "R_ARM" in b_set)
                or ("L_ARM" in b_set and "HEAD" in b_set)
                or ("R_ARM" in b_set and "HEAD" in b_set)
                or ("L_LEG" in b_set and "R_LEG" in b_set)
                or ("L_ARM" in b_set and ("L_LEG" in b_set or "R_LEG" in b_set))
                or ("R_ARM" in b_set and ("L_LEG" in b_set or "R_LEG" in b_set))
            ):
                faces_to_prune.append(f)
                reasons["kinematic"] += 1

    # 3. Prune tiny disconnected floating marching-cubes crumbs (< 35 faces) when a primary body mesh (> 2000 faces) exists
    pruned_set = set(faces_to_prune)
    visited_f: set[int] = set()
    islands: list[list] = []
    for f in bm.faces:
        if f in pruned_set or f.index in visited_f:
            continue
        isl = []
        stack = [f]
        visited_f.add(f.index)
        while stack:
            cur = stack.pop()
            isl.append(cur)
            for e in cur.edges:
                for lf in e.link_faces:
                    if lf not in pruned_set and lf.index not in visited_f:
                        visited_f.add(lf.index)
                        stack.append(lf)
        islands.append(isl)

    if islands and max(len(isl) for isl in islands) > 2000 and len(islands) > 15:
        for isl in islands:
            if len(isl) < 35:
                faces_to_prune.extend(isl)
                reasons["sliver_islands"] += len(isl)

    n_pruned = len(faces_to_prune)
    if n_pruned > 0 or n_welded > 0:
        if n_pruned > 0:
            log_fn(
                f"Pruning {n_pruned} topological artefact face(s) "
                f"({reasons['stretch']} metric stretch, {reasons['kinematic']} kinematic cross-limb bridges, "
                f"{reasons['sliver_islands']} floating sliver faces; "
                f"stretch > {stretch_thresh:.2f}, med_edge={med_edge:.3f})"
            )
            bmesh.ops.delete(bm, geom=faces_to_prune, context="FACES")

        wire_edges = [e for e in bm.edges if len(e.link_faces) == 0]
        if wire_edges:
            bmesh.ops.delete(bm, geom=wire_edges, context="EDGES")

        loose_verts = [v for v in bm.verts if len(v.link_edges) == 0]
        if loose_verts:
            bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

        bm.to_mesh(mesh_obj.data)
        mesh_obj.data.update()
        log_fn(
            f"Post-clean mesh: {len(mesh_obj.data.vertices)} vertices, {len(mesh_obj.data.polygons)} polygons"
        )
    else:
        log_fn(f"Topological webbing audit: 0 bridging artefacts detected (med_edge={med_edge:.3f})")

    bm.free()
    return n_pruned


def prepare_rig_in_blender(
    src_fbx: Path,
    out_fbx: Path,
    src_tex: Path | None = None,
    out_png: Path | None = None,
    log_fn: Callable[[str], None] = print,
) -> int:
    """Execute inside Blender (`bpy`) to convert `src_fbx` into a canonical `v2bm` FBX rig."""
    import bpy

    src_fbx = Path(src_fbx).resolve()
    out_fbx = Path(out_fbx).resolve()
    if src_tex is None:
        src_tex = discover_diffuse_texture(src_fbx)
    if out_png is None:
        out_png = out_fbx.with_suffix(".png")

    char_stem = out_fbx.stem
    log_fn(f"Importing source FBX from {src_fbx}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(src_fbx))

    arm_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if arm_obj is None or not mesh_objs:
        raise RuntimeError(f"Expected ARMATURE and MESH objects in {src_fbx}")
    # Pick the primary character mesh (highest vertex count)
    mesh_obj = max(mesh_objs, key=lambda o: len(o.data.vertices))
    empties = [o for o in bpy.data.objects if o.type == "EMPTY"]

    # Clear animation actions and switch armature to REST position
    arm_obj.animation_data_clear()
    for m in mesh_objs:
        m.animation_data_clear()
    for e in empties:
        e.animation_data_clear()
    arm_obj.data.pose_position = "REST"

    # Unparent from any scaled/rotated root Empty and reset transforms to identity
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
    surface_name = f"{char_stem}_Surface"
    mesh_obj.name = surface_name
    mesh_obj.data.name = surface_name

    # Auto-detect and strip any common bone/vertex-group prefix
    prefix = detect_common_bone_prefix([b.name for b in arm_obj.data.bones])
    if prefix:
        log_fn(f"Auto-detected common bone prefix: {prefix!r}")
        for b in arm_obj.data.bones:
            if b.name.startswith(prefix):
                b.name = b.name[len(prefix) :]
        for vg in mesh_obj.vertex_groups:
            if vg.name.startswith(prefix):
                vg.name = vg.name[len(prefix) :]

    # Audit and prune any topological webbing / bridging faces (fused contact geometry from 2D-to-3D reconstruction)
    prune_topological_webbing_in_blender(mesh_obj, log_fn)

    # If the imported mesh is excessively dense (> 80,000 polygons), apply scale-invariant,
    # facial-protected, UV-delimited decimation so skinning and ARKit morph targets stay fast and crisp
    n_polys = len(mesh_obj.data.polygons)
    if n_polys > 80000 and len(mesh_obj.data.vertices) > 0:
        xs = [v.co.x for v in mesh_obj.data.vertices]
        ys = [v.co.y for v in mesh_obj.data.vertices]
        zs = [v.co.z for v in mesh_obj.data.vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        w = max(max_x - min_x, 1e-6)
        h = max(max_y - min_y, 1e-6)
        d = max(max_z - min_z, 1e-6)
        cx = 0.5 * (min_x + max_x)
        cz = 0.5 * (min_z + max_z)

        vg_dec = mesh_obj.vertex_groups.new(name="__v2bm_DecimateWeight")
        for v in mesh_obj.data.vertices:
            co = v.co
            if (
                abs(co.x - cx) <= 0.14 * w
                and co.z >= cz + 0.08 * d
                and (max_y - 0.32 * h) <= co.y <= (max_y - 0.10 * h)
            ):
                vg_dec.add([v.index], 0.28, "REPLACE")
            elif co.y >= (max_y - 0.28 * h) and co.z < (cz + 0.06 * d):
                vg_dec.add([v.index], 1.0, "REPLACE")
            else:
                vg_dec.add([v.index], 0.55, "REPLACE")

        target_faces = 64000
        ratio = float(target_faces) / float(n_polys)
        log_fn(
            f"High-poly mesh ({n_polys} polygons > 80,000); applying UV-delimited decimation (ratio={ratio:.4f})..."
        )
        bpy.context.view_layer.objects.active = mesh_obj
        mod = mesh_obj.modifiers.new(name="CleanDecimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        mod.delimit = {"UV"}
        mod.vertex_group = "__v2bm_DecimateWeight"
        mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
        vg_rem = mesh_obj.vertex_groups.get("__v2bm_DecimateWeight")
        if vg_rem:
            mesh_obj.vertex_groups.remove(vg_rem)
        log_fn(
            f"Decimated mesh: {len(mesh_obj.data.vertices)} vertices, {len(mesh_obj.data.polygons)} polygons"
        )

    # Compute centering offset so hip is at X=0, Z=0 and soles touch Y=0
    n_verts = len(mesh_obj.data.vertices)
    coords = np.empty((n_verts, 3), dtype=np.float32)
    mesh_obj.data.vertices.foreach_get("co", coords.reshape(-1))

    hip_bone_name = next(
        (k for k in ("hip", "hips", "pelvis", "rig:Hips") if k in arm_obj.data.bones),
        arm_obj.data.bones[0].name,
    )
    hip_head = np.array(arm_obj.data.bones[hip_bone_name].head_local, dtype=np.float32)
    min_y = float(coords[:, 1].min())
    shift = np.array([-hip_head[0], -min_y, -hip_head[2]], dtype=np.float32)
    log_fn(f"Applying centering shift (cm): {shift.round(3).tolist()} (min_y={min_y:.3f})")

    coords = coords + shift[None, :]
    mesh_obj.data.vertices.foreach_set("co", coords.reshape(-1))
    mesh_obj.data.update()

    head_bone_name = next(
        (k for k in ("head", "Head", "rig:Head") if k in arm_obj.data.bones),
        None,
    )
    head_pos = (
        np.array(arm_obj.data.bones[head_bone_name].head_local, dtype=np.float32) + shift
        if head_bone_name is not None
        else None
    )

    # Ensure valid UV coordinates exist on the character mesh
    if not mesh_obj.data.uv_layers:
        log_fn("No UV coordinates detected; generating canonical Smart UV Project")
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.01)
        bpy.ops.object.mode_set(mode="OBJECT")

    # Build 52 ARKit shape keys before folding facial vertex groups into rig:Head
    build_arkit_shape_keys_in_blender(mesh_obj, coords, head_pos, log_fn)

    # Deduplicate canonical 61 bone order while preserving order
    canonical_61: list[str] = list(dict.fromkeys(BONE_RENAME_MAP.values()))
    weights_61: dict[str, np.ndarray] = {
        b: np.zeros(n_verts, dtype=np.float64) for b in canonical_61
    }

    for vg in list(mesh_obj.vertex_groups):
        old_name = vg.name
        old_low = old_name.lower()
        w_arr = extract_vg_weights(mesh_obj, old_name, n_verts).astype(np.float64)
        if old_name in canonical_61:
            weights_61[old_name] += w_arr
        elif old_low in BONE_RENAME_MAP:
            target_b = BONE_RENAME_MAP[old_low]
            weights_61[target_b] += w_arr
        elif old_low in WEIGHT_FOLD_MAP:
            target_b = WEIGHT_FOLD_MAP[old_low]
            weights_61[target_b] += w_arr
        elif old_low.endswith("_end_l") or old_low.endswith("_end_r"):
            side_sfx = "_l" if old_low.endswith("_l") else "_r"
            stem_pfx = old_low[: -len("_end_l")]
            cand = f"{stem_pfx}_03{side_sfx}"
            if cand in BONE_RENAME_MAP:
                weights_61[BONE_RENAME_MAP[cand]] += w_arr
            else:
                weights_61["rig:Head"] += w_arr
        else:
            log_fn(f"Warning: unmapped vertex group {old_name}, folding into rig:Hips")
            weights_61["rig:Hips"] += w_arr

    # Rigidly bind facial morph zone (anterior head above neck) 100% to rig:Head
    # so skeletal neck rotation never shears active facial morph deltas
    char_height = float(coords[:, 1].max() - coords[:, 1].min())
    if head_pos is not None:
        head_y_thresh = float(head_pos[1]) + 1.5
        head_x_rad = max(14.0, 0.08 * char_height)
        head_idx = canonical_61.index("rig:Head")
        weight_mat = np.column_stack([weights_61[b] for b in canonical_61])
        for i in range(n_verts):
            if (
                coords[i, 1] >= head_y_thresh
                and coords[i, 2] >= 1.5
                and abs(coords[i, 0] - head_pos[0]) <= head_x_rad
            ):
                weight_mat[i, :] = 0.0
                weight_mat[i, head_idx] = 1.0
    else:
        weight_mat = np.column_stack([weights_61[b] for b in canonical_61])

    # Clamp spurious distal bone weight bleeding (e.g. fingers/hands/feet influencing vertices
    # far away on the torso or head due to contacting limbs during 2D-to-3D neural skinning)
    clamped_entries = 0
    for b_idx, bname in enumerate(canonical_61):
        low = bname.lower()
        is_finger = any(k in low for k in ("thumb", "index", "middle", "ring", "pinky"))
        is_hand = "hand" in low and not is_finger
        is_foot = any(k in low for k in ("foot", "toe"))

        max_reach = None
        if is_finger:
            max_reach = max(18.0, 0.10 * char_height)
        elif is_hand:
            max_reach = max(26.0, 0.15 * char_height)
        elif is_foot:
            max_reach = max(30.0, 0.18 * char_height)

        raw_bname = bname.replace("rig:", "")
        bone = (
            arm_obj.data.bones.get(bname)
            or arm_obj.data.bones.get(raw_bname)
            or next((b for b in arm_obj.data.bones if BONE_RENAME_MAP.get(b.name.lower()) == bname), None)
        )

        if max_reach is not None and bone is not None:
            b_head = np.array(bone.head_local, dtype=np.float32) + shift
            dists = np.linalg.norm(coords - b_head[None, :], axis=1)
            bleed_mask = dists > max_reach
            n_bleed = int(np.count_nonzero(weight_mat[bleed_mask, b_idx] > 1e-4))
            if n_bleed > 0:
                weight_mat[bleed_mask, b_idx] = 0.0
                clamped_entries += n_bleed

    if clamped_entries > 0:
        log_fn(
            f"Clamped {clamped_entries} spurious distal weight bleeding assignment(s) across hands/fingers/feet"
        )

    # Normalise partition of unity across all canonical bones
    row_sums = weight_mat.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] < 1e-8
    if np.any(zero_rows):
        hips_idx = canonical_61.index("rig:Hips")
        weight_mat[zero_rows, hips_idx] = 1.0
        row_sums = weight_mat.sum(axis=1, keepdims=True)
    weight_mat /= row_sums
    max_unity_err = float(np.max(np.abs(weight_mat.sum(axis=1) - 1.0)))
    log_fn(
        f"Partition of unity max deviation across {len(canonical_61)} bones: {max_unity_err:.3e}"
    )

    # Replace vertex groups on mesh_obj with the exact canonical bone groups
    mesh_obj.vertex_groups.clear()
    for b_idx, bname in enumerate(canonical_61):
        vg = mesh_obj.vertex_groups.new(name=bname)
        col = weight_mat[:, b_idx]
        nz = np.where(col > 1e-5)[0]
        for v_idx in nz:
            vg.add([int(v_idx)], float(col[v_idx]), "REPLACE")

    # Edit armature: shift bone heads, rename canonical bones, remove non-canonical bones,
    # set uniform +Y edit bone tail (+5.0 cm) and zero roll for identity rest rotation
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_obj.data.edit_bones

    for b in eb:
        b.use_connect = False

    to_remove = [
        b for b in eb if b.name not in canonical_61 and b.name.lower() not in BONE_RENAME_MAP
    ]
    for b in to_remove:
        eb.remove(b)

    for b in list(eb):
        if b.name not in canonical_61:
            b.name = BONE_RENAME_MAP[b.name.lower()]

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

    if src_tex is not None and Path(src_tex).is_file():
        out_png.parent.mkdir(parents=True, exist_ok=True)
        bl_img = bpy.data.images.load(str(src_tex))
        if max(bl_img.size[0], bl_img.size[1]) > 4096:
            bl_img.scale(4096, 4096)
        bl_img.filepath_raw = str(out_png)
        bl_img.file_format = "PNG"
        bl_img.save()
        log_fn(
            f"Saved companion diffuse texture: {out_png} ({out_png.stat().st_size} bytes)"
        )
        bl_img.name = f"{char_stem}_diffuse.png"
        bl_img.pack()

        mat = bpy.data.materials.new(name=f"{char_stem}_Material")
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
    else:
        log_fn("No external diffuse texture found; retaining existing material setup")

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


def find_blender_executable(explicit: str | Path | None = None) -> Path:
    """Locate a working Blender executable for headless rig preparation."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified Blender executable not found: {p}")

    env_blender = os.environ.get("BLENDER_EXECUTABLE")
    if env_blender and Path(env_blender).is_file():
        return Path(env_blender)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        win_store = Path(local_app_data) / "Microsoft" / "WindowsApps" / "blender-launcher.exe"
        if win_store.exists():
            return win_store

    which_bl = shutil.which("blender") or shutil.which("blender-launcher")
    if which_bl:
        return Path(which_bl)

    raise FileNotFoundError(
        "Could not locate Blender executable. Install Blender 4.2+ or set BLENDER_EXECUTABLE."
    )


def prepare_rig(
    src_fbx: Path,
    out_fbx: Path,
    *,
    texture: Path | None = None,
    blender_exe: str | Path | None = None,
) -> Path:
    """Prepare a character FBX rig via headless Blender and verify it with `v2bm` checks."""
    src_fbx = Path(src_fbx).resolve()
    out_fbx = Path(out_fbx).resolve()
    if not src_fbx.is_file():
        raise FileNotFoundError(f"Source FBX not found: {src_fbx}")

    blender_bin = find_blender_executable(blender_exe)
    script_path = Path(__file__).resolve()
    cmd = [
        str(blender_bin),
        "--background",
        "--python",
        str(script_path),
        "--",
        "--input",
        str(src_fbx),
        "--output",
        str(out_fbx),
    ]
    if texture is not None:
        cmd.extend(["--texture", str(Path(texture).resolve())])

    print(f"Preparing rig via headless Blender: {src_fbx.name} -> {out_fbx}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log_path = out_fbx.with_suffix(".log")
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            print(f"  {line}")
        log_path.unlink(missing_ok=True)
    if proc.returncode != 0 or not out_fbx.is_file():
        raise RuntimeError(
            f"Headless Blender rig preparation failed (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )

    from .. import checks

    problem = checks.check_rig(out_fbx)
    if problem is not None:
        raise RuntimeError(f"Prepared rig {out_fbx} failed validation: {problem}")
    print(f"[ok]      Prepared rig verified: {out_fbx}")
    return out_fbx


def _blender_cli_main() -> int:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Headless Blender rig preparation entry point")
    parser.add_argument("--input", required=True, help="Source FBX path")
    parser.add_argument("--output", required=True, help="Output canonical FBX path")
    parser.add_argument("--texture", default=None, help="Optional diffuse texture path")
    args = parser.parse_args(argv)

    src_fbx = Path(args.input).resolve()
    out_fbx = Path(args.output).resolve()
    src_tex = Path(args.texture).resolve() if args.texture else None
    out_png = out_fbx.with_suffix(".png")
    log_path = out_fbx.with_suffix(".log")

    if log_path.exists():
        log_path.unlink()

    def log_fn(msg: str) -> None:
        print(msg)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        return prepare_rig_in_blender(src_fbx, out_fbx, src_tex=src_tex, out_png=out_png, log_fn=log_fn)
    except Exception:
        import traceback

        log_fn("ERROR:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(_blender_cli_main())

"""Humanoid Skeleton motion → Humanoid Rig Rig motion, derived from the T-pose calibration.

    validated_humanoid_motion
            ↓
    tpose_calibration
            ↓
    apply_rest_relative_rotations()
            ↓
    target_rig_fk

For every controlled bone we measure how the Humanoid Skeleton bone moved away
from its own rest/T-pose, and apply that same rest-relative motion to the
corresponding Humanoid Rig bone:

    delta_skeleton   = inverse(skeleton_rest_local_rotation) * skeleton_frame_local_rotation
    humanoid_rig_frame_local = humanoid_rig_rest_local_rotation * (C * delta_skeleton * Cᵀ)

`C` is the per-bone rest-frame basis conversion. It is *derived* from the two
validated T-pose bone frames (see `bone_basis_change`), never authored by hand.
No world-space Euler angles are copied, and there are no per-bone 90°/180°
constants, laterality swaps or camera-based assumptions anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kinematics import HumanoidRig, forward_kinematics, world_positions_m
from .skinned_mesh import load_full_rig
from .tpose_calibration import (
    TposeCalibration,
    anatomical_axes,
    global_alignment,
    hips_origin,
)

# Bones whose motion is authored by the validated Human → Humanoid retarget.
# Anatomical LEFT stays LEFT and RIGHT stays RIGHT.
CONTROLLED_BONES = (
    "rig:Hips",
    "rig:Spine",
    "rig:Spine1",
    "rig:Spine2",
    "rig:Neck",
    "rig:Head",
    "rig:LeftShoulder",
    "rig:LeftArm",
    "rig:LeftForeArm",
    "rig:LeftHand",
    "rig:RightShoulder",
    "rig:RightArm",
    "rig:RightForeArm",
    "rig:RightHand",
    "rig:LeftUpLeg",
    "rig:LeftLeg",
    "rig:LeftFoot",
    "rig:LeftToeBase",
    "rig:RightUpLeg",
    "rig:RightLeg",
    "rig:RightFoot",
    "rig:RightToeBase",
)


def rest_world_rotations(rig: HumanoidRig) -> dict[str, np.ndarray]:
    """World rotation of each bone's rest frame.

    FK is `M_local = T * Rpre * R_anim`, so at rest (`R_anim = I`) the bone's
    world rotation is `R_parent_world @ Rpre`. PreRotation therefore stays part
    of the rest transform and is never folded into the animation channel.
    """
    world = forward_kinematics(rig, {})
    return {name: np.asarray(mat[:3, :3], dtype=np.float64) for name, mat in world.items()}


def bone_basis_change(
    skel_rest_rot: dict[str, np.ndarray],
    humanoid_rig_rest_rot: dict[str, np.ndarray],
    bone: str,
    r_world_align: np.ndarray,
) -> np.ndarray:
    """Rest-frame basis conversion `C` for one bone, from the two T-pose frames.

    A rest-relative local rotation `D` acts on the bone's own rest frame. In the
    Humanoid Skeleton that frame sits in the world as `F = skel_rest_rot[bone]`;
    in the Humanoid Rig as `F' = humanoid_rig_rest_rot[bone]`. Requiring the *world* motion to
    match (up to the calibrated world alignment `R`) gives

        F' D' F'ᵀ = R (F D Fᵀ) Rᵀ        ⇒        D' = C D Cᵀ,   C = F'ᵀ R F

    so `C` falls straight out of the two validated rest poses.
    """
    f_skel = np.asarray(skel_rest_rot[bone], dtype=np.float64)
    f_humanoid_rig = np.asarray(humanoid_rig_rest_rot[bone], dtype=np.float64)
    return f_humanoid_rig.T @ np.asarray(r_world_align, dtype=np.float64) @ f_skel


@dataclass(frozen=True)
class RigMapping:
    """Frozen Humanoid Skeleton → Humanoid Rig transfer, built once from the calibration."""

    skeleton_rig: HumanoidRig
    target_rig: HumanoidRig
    basis_change: dict[str, np.ndarray]
    translation_scale: float
    r_world_align: np.ndarray
    max_basis_deviation_deg: float

    @property
    def controlled(self) -> tuple[str, ...]:
        return tuple(name for name in CONTROLLED_BONES if name in self.target_rig.bones)

    def apply_rest_relative_rotations(
        self, r_anim_skeleton: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Transfer one frame of rest-relative local rotations onto the Humanoid Rig rig.

        Input and output are both animation-channel rotations, i.e. the `R_anim`
        of `M_local = T * Rpre * R_anim`. Bones the Humanoid retarget does not
        author (the finger chains) are left at rest.
        """
        out: dict[str, np.ndarray] = {}
        for bone, delta in r_anim_skeleton.items():
            change = self.basis_change.get(bone)
            if change is None:
                continue
            out[bone] = change @ np.asarray(delta, dtype=np.float64) @ change.T
        return out

    def hips_translation_cm(self, pelvis_translation_cm: np.ndarray) -> np.ndarray:
        """Root translation carried across with the calibrated scale and world alignment."""
        p = np.asarray(pelvis_translation_cm, dtype=np.float64).reshape(3)
        return self.translation_scale * (self.r_world_align @ p)

    def target_rig_fk(
        self,
        r_anim_humanoid_rig: dict[str, np.ndarray],
        hips_translation_cm: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """World 4x4 per Humanoid Rig bone. Raw FBX rest hierarchy, animation on top."""
        return forward_kinematics(
            self.target_rig,
            r_anim_humanoid_rig,
            hips_translation_cm=np.asarray(hips_translation_cm, dtype=np.float64),
        )

    def humanoid_rig_positions_m(
        self,
        r_anim_humanoid_rig: dict[str, np.ndarray],
        hips_translation_cm: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return world_positions_m(self.target_rig, self.target_rig_fk(r_anim_humanoid_rig, hips_translation_cm))


def _rotation_angle_deg(rot: np.ndarray) -> float:
    cosine = (float(np.trace(np.asarray(rot, dtype=np.float64))) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def build_mapping(
    calibration: TposeCalibration,
    fbx_path: Path | None = None,
    *,
    verbose: bool = True,
) -> RigMapping:
    """Derive the Humanoid Skeleton → character-rig transfer from the calibration."""
    fbx_path = Path(fbx_path) if fbx_path is not None else calibration.fbx_path
    skel_rig = calibration.skeleton_rig
    target_rig = load_full_rig(fbx_path)

    skel_rest_rot = rest_world_rotations(skel_rig)
    humanoid_rig_rest_rot = rest_world_rotations(target_rig)

    # World alignment between the two rigs, measured with the same
    # `anatomical_axes` / `global_alignment` that produced the accepted
    # orientation-check images — so the animation cannot invent a different
    # Front/Right/Up convention. Both sides are the rigs' own rest poses.
    skel_rest_pos = hips_origin(world_positions_m(skel_rig, forward_kinematics(skel_rig, {})))
    humanoid_rig_rest_pos = hips_origin(world_positions_m(target_rig, forward_kinematics(target_rig, {})))
    skel_rest_axes = anatomical_axes(skel_rest_pos, "Humanoid Skeleton rest", verbose=False)
    humanoid_rig_rest_axes = anatomical_axes(humanoid_rig_rest_pos, "Humanoid Rig rig rest", verbose=False)
    r_world_align = global_alignment(skel_rest_axes, humanoid_rig_rest_axes, verbose=False)["R"]

    # The Humanoid Rig anatomical frame used here must be the one the calibration
    # accepted; otherwise the render camera and the mapping would disagree.
    front_drift = float(
        np.degrees(
            np.arccos(
                np.clip(float(np.asarray(humanoid_rig_rest_axes["forward"]) @ calibration.forward), -1.0, 1.0)
            )
        )
    )
    if front_drift > 1e-6:
        raise RuntimeError(
            f"rig FRONT disagrees with the accepted calibration by {front_drift:.6f} deg"
        )

    basis_change: dict[str, np.ndarray] = {}
    worst = 0.0
    for bone in CONTROLLED_BONES:
        if bone not in skel_rest_rot or bone not in humanoid_rig_rest_rot:
            continue
        change = bone_basis_change(skel_rest_rot, humanoid_rig_rest_rot, bone, r_world_align)
        basis_change[bone] = change
        worst = max(worst, _rotation_angle_deg(change))

    translation_scale = float(skel_rig.meters_per_unit / target_rig.meters_per_unit)

    if verbose:
        print("\n=== Humanoid Skeleton -> Humanoid Rig rest-relative mapping ===")
        print(f"  skeleton rig            {skel_rig.source.name}  {len(skel_rig.order)} bones")
        print(f"  Humanoid Rig rig               {Path(fbx_path).name}  {len(target_rig.order)} bones")
        print(f"  controlled bones        {len(basis_change)}")
        print(f"  untouched (rest) bones  {len(target_rig.order) - len(basis_change)}  (finger chains)")
        print(f"  translation scale       {translation_scale:.6f}")
        print(f"  world alignment         {_rotation_angle_deg(r_world_align):.6f} deg  (measured, det={np.linalg.det(r_world_align):+.6f})")
        print(f"  max |C| rotation        {worst:.6f} deg   (0 = identical rest frames)")
        if worst < 1e-6:
            print("  the two validated T-pose bone frames coincide, so C = I for every bone:")
            print("  the Humanoid rest-relative rotation is applied to the Humanoid Rig unchanged.")
        else:
            print("  C is a measured rest-frame basis change, not a hand-authored offset.")

    return RigMapping(
        skeleton_rig=skel_rig,
        target_rig=target_rig,
        basis_change=basis_change,
        translation_scale=translation_scale,
        r_world_align=r_world_align,
        max_basis_deviation_deg=worst,
    )

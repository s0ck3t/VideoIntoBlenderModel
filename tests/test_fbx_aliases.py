"""Unit tests for humanoid bone name normalisation and rig validation.

Verifies two-stage bone normalisation across all standard third-party humanoid conventions:
- Mixamo namespace prefixes (`mixamorig:`, `mixamorig1:`)
- Unprefixed bare humanoid names (`Hips`, `Spine`, `Chest`, `LeftArm`)
- Quaternius / Unreal naming (`pelvis`, `spine_01`, `upperarm_l`, `thigh_l`)
- Kenney naming (`Chest`, `UpperChest`, `LeftToes`, `RightToes`)
- MakeHuman / MPFB2 naming (`spine01`, `neck01`, `toe_l`)
- VRM / Unity Humanoid conventions (`J_Bip_C_Hips`, `J_Bip_L_UpperArm`)
- 3ds Max Biped conventions (`Bip01_Pelvis`, `Bip01_L_UpperArm`)
- Exact preservation of canonical `rig:*` bones (100% backwards compatibility)
- Integration with `v2bm.checks.check_rig`
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest

from v2bm import checks
from v2bm.rig.fbx_skeleton import (
    ALL_HUMANOID_BONE_SUFFIXES,
    CORE_BONE_SUFFIXES,
    FINGER_BONE_SUFFIXES,
    HUMANOID_BONE_ALIASES,
    TWIST_BONE_SUFFIXES,
    bone_suffix,
    canonicalize_bone_name,
    is_core_bone,
    is_humanoid_bone,
    normalize_rig_prefix,
)
from v2bm.rig.rig_retarget import CONTROLLED_BONES, CORE_CONTROLLED_BONES, RigMapping

REPO_ROOT = Path(__file__).resolve().parent.parent
HUMANOID_RIG_FBX = REPO_ROOT / "assets" / "rigs" / "humanoid_rig.fbx"


def test_canonical_rig_bones_preserved() -> None:
    """Canonical `rig:<Suffix>` bone names must be preserved verbatim."""
    for suffix in CORE_BONE_SUFFIXES:
        canonical = f"rig:{suffix}"
        assert canonicalize_bone_name(canonical) == canonical
        assert normalize_rig_prefix(canonical) == canonical
        assert is_core_bone(canonical) is True
        assert bone_suffix(canonical) == suffix


def test_numerical_rig_prefixes() -> None:
    """`rig\\d+:<Suffix>` bones (from multi-character exports) normalise to `rig:<Suffix>`."""
    assert canonicalize_bone_name("rig0:Hips") == "rig:Hips"
    assert canonicalize_bone_name("rig1:Spine") == "rig:Spine"
    assert canonicalize_bone_name("rig8:LeftArm") == "rig:LeftArm"
    assert normalize_rig_prefix("rig8:RightLeg") == "rig:RightLeg"


def test_mixamo_prefix_normalisation() -> None:
    """Mixamo prefixes (`mixamorig:`, `mixamorig1:`) normalise cleanly to canonical `rig:*`."""
    mixamo_cases = {
        "mixamorig:Hips": "rig:Hips",
        "mixamorig1:Hips": "rig:Hips",
        "mixamorig9:Hips": "rig:Hips",
        "mixamorig:Spine": "rig:Spine",
        "mixamorig:Spine1": "rig:Spine1",
        "mixamorig:Spine2": "rig:Spine2",
        "mixamorig:Neck": "rig:Neck",
        "mixamorig:Head": "rig:Head",
        "mixamorig:HeadTop_End": "rig:HeadTop_End",
        "mixamorig:LeftShoulder": "rig:LeftShoulder",
        "mixamorig:LeftArm": "rig:LeftArm",
        "mixamorig:LeftForeArm": "rig:LeftForeArm",
        "mixamorig:LeftHand": "rig:LeftHand",
        "mixamorig:RightShoulder": "rig:RightShoulder",
        "mixamorig:RightArm": "rig:RightArm",
        "mixamorig:RightForeArm": "rig:RightForeArm",
        "mixamorig:RightHand": "rig:RightHand",
        "mixamorig:LeftUpLeg": "rig:LeftUpLeg",
        "mixamorig:LeftLeg": "rig:LeftLeg",
        "mixamorig:LeftFoot": "rig:LeftFoot",
        "mixamorig:LeftToeBase": "rig:LeftToeBase",
        "mixamorig:LeftToe_End": "rig:LeftToe_End",
        "mixamorig:RightUpLeg": "rig:RightUpLeg",
        "mixamorig:RightLeg": "rig:RightLeg",
        "mixamorig:RightFoot": "rig:RightFoot",
        "mixamorig:RightToeBase": "rig:RightToeBase",
        "mixamorig:RightToe_End": "rig:RightToe_End",
    }
    for raw, expected in mixamo_cases.items():
        assert canonicalize_bone_name(raw) == expected
        assert normalize_rig_prefix(raw) == expected
        assert is_core_bone(raw) is True
        assert bone_suffix(raw) == expected.split(":")[-1]


def test_bare_humanoid_names() -> None:
    """Bare unprefixed standard humanoid bone names normalise to canonical `rig:*`."""
    for suffix in CORE_BONE_SUFFIXES:
        assert canonicalize_bone_name(suffix) == f"rig:{suffix}"
        assert normalize_rig_prefix(suffix) == f"rig:{suffix}"
        assert is_core_bone(suffix) is True
        assert bone_suffix(suffix) == suffix


def test_quaternius_unreal_conventions() -> None:
    """Quaternius and Unreal Engine nomenclature normalises correctly."""
    quaternius_cases = {
        "pelvis": "rig:Hips",
        "root": "rig:Hips",
        "spine_01": "rig:Spine",
        "spine_02": "rig:Spine1",
        "spine_03": "rig:Spine2",
        "clavicle_l": "rig:LeftShoulder",
        "upperarm_l": "rig:LeftArm",
        "lowerarm_l": "rig:LeftForeArm",
        "hand_l": "rig:LeftHand",
        "thigh_l": "rig:LeftUpLeg",
        "calf_l": "rig:LeftLeg",
        "foot_l": "rig:LeftFoot",
        "ball_l": "rig:LeftToeBase",
        "clavicle_r": "rig:RightShoulder",
        "upperarm_r": "rig:RightArm",
        "lowerarm_r": "rig:RightForeArm",
        "hand_r": "rig:RightHand",
        "thigh_r": "rig:RightUpLeg",
        "calf_r": "rig:RightLeg",
        "foot_r": "rig:RightFoot",
        "ball_r": "rig:RightToeBase",
    }
    for raw, expected in quaternius_cases.items():
        assert canonicalize_bone_name(raw) == expected
        assert is_core_bone(raw) is True
        assert bone_suffix(raw) == expected.split(":")[-1]


def test_kenney_conventions() -> None:
    """Kenney humanoid bone conventions normalise correctly."""
    kenney_cases = {
        "Chest": "rig:Spine1",
        "UpperChest": "rig:Spine2",
        "LeftToes": "rig:LeftToeBase",
        "RightToes": "rig:RightToeBase",
        "lefttoes": "rig:LeftToeBase",
        "righttoes": "rig:RightToeBase",
    }
    for raw, expected in kenney_cases.items():
        assert canonicalize_bone_name(raw) == expected
        assert is_core_bone(raw) is True
        assert bone_suffix(raw) == expected.split(":")[-1]


def test_makehuman_mpfb2_conventions() -> None:
    """MakeHuman / MPFB2 humanoid conventions normalise correctly."""
    makehuman_cases = {
        "spine01": "rig:Spine",
        "spine02": "rig:Spine1",
        "spine03": "rig:Spine2",
        "neck01": "rig:Neck",
        "head_end": "rig:HeadTop_End",
        "headtop_end": "rig:HeadTop_End",
        "toe_l": "rig:LeftToeBase",
        "toe_r": "rig:RightToeBase",
        "wrist_l": "rig:LeftHand",
        "wrist_r": "rig:RightHand",
        "shin_l": "rig:LeftLeg",
        "shin_r": "rig:RightLeg",
    }
    for raw, expected in makehuman_cases.items():
        assert canonicalize_bone_name(raw) == expected
        assert is_core_bone(raw) is True


def test_vrm_unity_conventions() -> None:
    """VRM and Unity Humanoid naming conventions normalise correctly."""
    vrm_cases = {
        "J_Bip_C_Hips": "rig:Hips",
        "J_Bip_C_Spine": "rig:Spine",
        "J_Bip_C_Chest": "rig:Spine1",
        "J_Bip_C_UpperChest": "rig:Spine2",
        "J_Bip_C_Neck": "rig:Neck",
        "J_Bip_C_Head": "rig:Head",
        "J_Bip_L_Shoulder": "rig:LeftShoulder",
        "J_Bip_L_UpperArm": "rig:LeftArm",
        "J_Bip_L_LowerArm": "rig:LeftForeArm",
        "J_Bip_L_Hand": "rig:LeftHand",
        "J_Bip_L_UpperLeg": "rig:LeftUpLeg",
        "J_Bip_L_LowerLeg": "rig:LeftLeg",
        "J_Bip_L_Foot": "rig:LeftFoot",
        "J_Bip_L_ToeBase": "rig:LeftToeBase",
        "J_Bip_R_Shoulder": "rig:RightShoulder",
        "J_Bip_R_UpperArm": "rig:RightArm",
        "J_Bip_R_LowerArm": "rig:RightForeArm",
        "J_Bip_R_Hand": "rig:RightHand",
        "J_Bip_R_UpperLeg": "rig:RightUpLeg",
        "J_Bip_R_LowerLeg": "rig:RightLeg",
        "J_Bip_R_Foot": "rig:RightFoot",
        "J_Bip_R_ToeBase": "rig:RightToeBase",
    }
    for raw, expected in vrm_cases.items():
        assert canonicalize_bone_name(raw) == expected
        assert is_core_bone(raw) is True


def test_3dsmax_biped_conventions() -> None:
    """3ds Max Biped bone conventions normalise correctly."""
    biped_cases = {
        "Bip01_Pelvis": "rig:Hips",
        "Bip01_Spine": "rig:Spine",
        "Bip01_Spine1": "rig:Spine1",
        "Bip01_Spine2": "rig:Spine2",
        "Bip01_Neck": "rig:Neck",
        "Bip01_Head": "rig:Head",
        "Bip01_L_Clavicle": "rig:LeftShoulder",
        "Bip01_L_UpperArm": "rig:LeftArm",
        "Bip01_L_Forearm": "rig:LeftForeArm",
        "Bip01_L_Hand": "rig:LeftHand",
        "Bip01_L_Thigh": "rig:LeftUpLeg",
        "Bip01_L_Calf": "rig:LeftLeg",
        "Bip01_L_Foot": "rig:LeftFoot",
        "Bip01_L_Toe0": "rig:LeftToeBase",
        "Bip01_R_Clavicle": "rig:RightShoulder",
        "Bip01_R_UpperArm": "rig:RightArm",
        "Bip01_R_Forearm": "rig:RightForeArm",
        "Bip01_R_Hand": "rig:RightHand",
        "Bip01_R_Thigh": "rig:RightUpLeg",
        "Bip01_R_Calf": "rig:RightLeg",
        "Bip01_R_Foot": "rig:RightFoot",
        "Bip01_R_Toe0": "rig:RightToeBase",
    }
    for raw, expected in biped_cases.items():
        assert canonicalize_bone_name(raw) == expected
        assert is_core_bone(raw) is True


def test_blender_dot_notation() -> None:
    """Blender dot notation (e.g. `upperarm.l`) normalises correctly."""
    dot_cases = {
        "shoulder.l": "rig:LeftShoulder",
        "upperarm.l": "rig:LeftArm",
        "lowerarm.l": "rig:LeftForeArm",
        "hand.l": "rig:LeftHand",
        "thigh.l": "rig:LeftUpLeg",
        "calf.l": "rig:LeftLeg",
        "foot.l": "rig:LeftFoot",
        "toe.l": "rig:LeftToeBase",
        "shoulder.r": "rig:RightShoulder",
        "upperarm.r": "rig:RightArm",
        "lowerarm.r": "rig:RightForeArm",
        "hand.r": "rig:RightHand",
        "thigh.r": "rig:RightUpLeg",
        "calf.r": "rig:RightLeg",
        "foot.r": "rig:RightFoot",
        "toe.r": "rig:RightToeBase",
    }
    for raw, expected in dot_cases.items():
        assert canonicalize_bone_name(raw) == expected
        assert is_core_bone(raw) is True


def test_case_insensitivity() -> None:
    """Bone normalisation is case-insensitive."""
    assert canonicalize_bone_name("HIPS") == "rig:Hips"
    assert canonicalize_bone_name("hips") == "rig:Hips"
    assert canonicalize_bone_name("leftarm") == "rig:LeftArm"
    assert canonicalize_bone_name("LEFTARM") == "rig:LeftArm"
    assert canonicalize_bone_name("UPPERARM_L") == "rig:LeftArm"
    assert canonicalize_bone_name("pElViS") == "rig:Hips"
    assert canonicalize_bone_name("mIxAmOrIg:LeFtArM") == "rig:LeftArm"


def test_non_core_limbs_retained() -> None:
    """Non-humanoid nodes (e.g. props, accessories, cameras) are retained without false humanoid mapping."""
    assert canonicalize_bone_name("mixamorig:Sword_Prop") == "Sword_Prop"
    assert is_core_bone("Sword_Prop") is False
    assert is_humanoid_bone("Sword_Prop") is False
    assert canonicalize_bone_name("Camera") == "Camera"
    assert is_core_bone("Camera") is False
    assert is_humanoid_bone("Camera") is False
    assert canonicalize_bone_name("bip01_weapon") == "weapon"
    assert is_core_bone("weapon") is False
    assert is_humanoid_bone("weapon") is False


def test_check_rig_with_humanoid_rig_asset() -> None:
    """Verify check_rig passes on humanoid_rig.fbx."""
    if not HUMANOID_RIG_FBX.exists():
        pytest.skip(f"Reference rig not found at {HUMANOID_RIG_FBX}")
    result = checks.check_rig(HUMANOID_RIG_FBX)
    assert result is None, f"Expected check_rig to pass, got error: {result}"


def test_check_rig_validates_normalised_skeleton_conventions(tmp_path: Path) -> None:
    """Verify check_rig validates normalised third-party skeleton bone dictionaries."""
    dummy_fbx = tmp_path / "dummy.fbx"
    dummy_fbx.write_bytes(checks.FBX_BINARY_MAGIC + b"\x00" * 100)

    # 1. Mock skeleton returning third-party Mixamo bone names
    mixamo_bones = [
        {"name": canonicalize_bone_name(b)}
        for b in [
            "mixamorig:Hips",
            "mixamorig:Spine",
            "mixamorig:Spine1",
            "mixamorig:LeftArm",
            "mixamorig:RightArm",
            "mixamorig:LeftUpLeg",
            "mixamorig:RightUpLeg",
        ]
    ]
    with patch("v2bm.rig.fbx_skeleton.extract_skeleton", return_value={"bones": mixamo_bones}):
        res = checks.check_rig(dummy_fbx)
        assert res is None, f"Expected Mixamo normalised rig to pass, got: {res}"

    # 2. Mock skeleton returning bare bone names
    bare_bones = [
        {"name": canonicalize_bone_name(b)}
        for b in ["Hips", "Spine", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]
    ]
    with patch("v2bm.rig.fbx_skeleton.extract_skeleton", return_value={"bones": bare_bones}):
        res = checks.check_rig(dummy_fbx)
        assert res is None, f"Expected bare normalised rig to pass, got: {res}"

    # 3. Mock skeleton missing Hips
    missing_hips = [
        {"name": canonicalize_bone_name(b)}
        for b in ["Spine", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]
    ]
    with patch("v2bm.rig.fbx_skeleton.extract_skeleton", return_value={"bones": missing_hips}):
        res = checks.check_rig(dummy_fbx)
        assert res is not None
        assert "has no Humanoid skeleton" in res

    # 4. Mock skeleton missing essential limb (e.g. LeftArm)
    missing_limb = [
        {"name": canonicalize_bone_name(b)}
        for b in ["Hips", "Spine", "RightArm", "LeftUpLeg", "RightUpLeg"]
    ]
    with patch("v2bm.rig.fbx_skeleton.extract_skeleton", return_value={"bones": missing_limb}):
        res = checks.check_rig(dummy_fbx)
        assert res is not None
        assert "is missing essential Humanoid bones" in res
        assert "rig:LeftArm" in res


def test_unit_to_meters_runtime_auto_calibration() -> None:
    """Verify unit_to_meters auto-calibrates arbitrary third-party rig scales at runtime."""
    import numpy as np
    from v2bm.rig.fbx_skeleton import CANONICAL_HUMANOID_SPAN_M, unit_to_meters

    # 1. Standard centimetre rig (172 units with UnitScaleFactor=1.0 -> 1.72 m) keeps 0.01 exactly
    cm_pts = [np.array([0.0, 3.0, 0.0]), np.array([0.0, 175.0, 0.0])]
    assert np.isclose(unit_to_meters({"UnitScaleFactor": 1.0}, cm_pts), 0.01)

    # 2. Raw game-unit rig (3.76 units with UnitScaleFactor=1.0 -> 0.0376 m) auto-calibrates to 1.70 m
    game_pts = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 3.76, 0.0])]
    mpu_game = unit_to_meters({"UnitScaleFactor": 1.0}, game_pts)
    assert np.isclose(3.76 * mpu_game, CANONICAL_HUMANOID_SPAN_M)

    # 3. Metre-authored rig (1.75 units with uncalibrated UnitScaleFactor=1.0) auto-calibrates to 1.70 m
    m_pts = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.75, 0.0])]
    mpu_m = unit_to_meters({"UnitScaleFactor": 1.0}, m_pts)
    assert np.isclose(1.75 * mpu_m, CANONICAL_HUMANOID_SPAN_M)

    # 4. Millimetre-authored rig (1750 units with UnitScaleFactor=1.0) auto-calibrates to 1.70 m
    mm_pts = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1750.0, 0.0])]
    mpu_mm = unit_to_meters({"UnitScaleFactor": 1.0}, mm_pts)
    assert np.isclose(1750.0 * mpu_mm, CANONICAL_HUMANOID_SPAN_M)


def test_finger_bone_aliases() -> None:
    """Verify 30 phalangeal finger bones across Mixamo, Blender dot-notation, VRM/Unity, MakeHuman, Kenney, and 3ds Max."""
    # Mixamo
    for s in FINGER_BONE_SUFFIXES:
        raw = f"mixamorig:{s}"
        assert canonicalize_bone_name(raw) == f"rig:{s}"
        assert is_humanoid_bone(raw) is True
        assert is_core_bone(raw) is False

    # Blender dot-notation
    blender_fingers = {
        "thumb.01.l": "rig:LeftHandThumb1",
        "thumb.02.l": "rig:LeftHandThumb2",
        "thumb.03.l": "rig:LeftHandThumb3",
        "f_index.01.l": "rig:LeftHandIndex1",
        "index.02.l": "rig:LeftHandIndex2",
        "index.03.l": "rig:LeftHandIndex3",
        "middle.01.l": "rig:LeftHandMiddle1",
        "middle.02.l": "rig:LeftHandMiddle2",
        "middle.03.l": "rig:LeftHandMiddle3",
        "ring.01.l": "rig:LeftHandRing1",
        "ring.02.l": "rig:LeftHandRing2",
        "ring.03.l": "rig:LeftHandRing3",
        "pinky.01.l": "rig:LeftHandPinky1",
        "little.02.l": "rig:LeftHandPinky2",
        "pinky.03.l": "rig:LeftHandPinky3",
        "thumb.01.r": "rig:RightHandThumb1",
        "index.01.r": "rig:RightHandIndex1",
        "middle.01.r": "rig:RightHandMiddle1",
        "ring.01.r": "rig:RightHandRing1",
        "pinky.01.r": "rig:RightHandPinky1",
    }
    for raw, expected in blender_fingers.items():
        assert canonicalize_bone_name(raw) == expected

    # VRM / Unity Humanoid
    vrm_fingers = {
        "leftThumbProximal": "rig:LeftHandThumb1",
        "leftThumbIntermediate": "rig:LeftHandThumb2",
        "leftThumbDistal": "rig:LeftHandThumb3",
        "leftIndexProximal": "rig:LeftHandIndex1",
        "leftIndexIntermediate": "rig:LeftHandIndex2",
        "leftIndexDistal": "rig:LeftHandIndex3",
        "leftMiddleProximal": "rig:LeftHandMiddle1",
        "leftRingProximal": "rig:LeftHandRing1",
        "leftLittleProximal": "rig:LeftHandPinky1",
        "rightThumbProximal": "rig:RightHandThumb1",
        "rightIndexProximal": "rig:RightHandIndex1",
        "rightMiddleProximal": "rig:RightHandMiddle1",
        "rightRingProximal": "rig:RightHandRing1",
        "rightLittleProximal": "rig:RightHandPinky1",
        "J_Bip_L_Thumb1": "rig:LeftHandThumb1",
        "J_Bip_L_Index1": "rig:LeftHandIndex1",
        "J_Bip_R_Thumb1": "rig:RightHandThumb1",
    }
    for raw, expected in vrm_fingers.items():
        assert canonicalize_bone_name(raw) == expected

    # MakeHuman / MPFB2
    makehuman_fingers = {
        "finger1.1.l": "rig:LeftHandThumb1",
        "finger1.2.l": "rig:LeftHandThumb2",
        "finger1.3.l": "rig:LeftHandThumb3",
        "finger2.1.l": "rig:LeftHandIndex1",
        "finger3.1.l": "rig:LeftHandMiddle1",
        "finger4.1.l": "rig:LeftHandRing1",
        "finger5.1.l": "rig:LeftHandPinky1",
        "finger1.1.r": "rig:RightHandThumb1",
        "finger2.1.r": "rig:RightHandIndex1",
        "finger3.1.r": "rig:RightHandMiddle1",
        "finger4.1.r": "rig:RightHandRing1",
        "finger5.1.r": "rig:RightHandPinky1",
    }
    for raw, expected in makehuman_fingers.items():
        assert canonicalize_bone_name(raw) == expected

    # 3ds Max Biped
    biped_fingers = {
        "Bip01_L_Finger0": "rig:LeftHandThumb1",
        "Bip01_L_Finger01": "rig:LeftHandThumb2",
        "Bip01_L_Finger02": "rig:LeftHandThumb3",
        "Bip01_L_Finger1": "rig:LeftHandIndex1",
        "Bip01_L_Finger2": "rig:LeftHandMiddle1",
        "Bip01_L_Finger3": "rig:LeftHandRing1",
        "Bip01_L_Finger4": "rig:LeftHandPinky1",
        "Bip01_R_Finger0": "rig:RightHandThumb1",
        "Bip01_R_Finger1": "rig:RightHandIndex1",
        "Bip01_R_Finger2": "rig:RightHandMiddle1",
        "Bip01_R_Finger3": "rig:RightHandRing1",
        "Bip01_R_Finger4": "rig:RightHandPinky1",
    }
    for raw, expected in biped_fingers.items():
        assert canonicalize_bone_name(raw) == expected


def test_limb_twist_aliases() -> None:
    """Verify 6 limb-twist bones across Mixamo, Blender, VRM, and Biped conventions."""
    twist_cases = {
        "mixamorig:LeftArmTwist": "rig:LeftArmTwist",
        "arm_twist.l": "rig:LeftArmTwist",
        "upperarm_twist_l": "rig:LeftArmTwist",
        "bip01_l_armtwist": "rig:LeftArmTwist",
        "j_bip_l_armtwist": "rig:LeftArmTwist",

        "mixamorig:RightArmTwist": "rig:RightArmTwist",
        "arm_twist.r": "rig:RightArmTwist",
        "upperarm_twist_r": "rig:RightArmTwist",
        "bip01_r_armtwist": "rig:RightArmTwist",

        "mixamorig:LeftForeArmTwist": "rig:LeftForeArmTwist",
        "forearm_twist.l": "rig:LeftForeArmTwist",
        "lowerarm_twist_l": "rig:LeftForeArmTwist",
        "bip01_l_forearmtwist": "rig:LeftForeArmTwist",

        "mixamorig:RightForeArmTwist": "rig:RightForeArmTwist",
        "forearm_twist.r": "rig:RightForeArmTwist",
        "lowerarm_twist_r": "rig:RightForeArmTwist",
        "bip01_r_forearmtwist": "rig:RightForeArmTwist",

        "mixamorig:LeftUpLegTwist": "rig:LeftUpLegTwist",
        "thigh_twist.l": "rig:LeftUpLegTwist",
        "upleg_twist_l": "rig:LeftUpLegTwist",
        "bip01_l_thightwist": "rig:LeftUpLegTwist",

        "mixamorig:RightUpLegTwist": "rig:RightUpLegTwist",
        "thigh_twist.r": "rig:RightUpLegTwist",
        "upleg_twist_r": "rig:RightUpLegTwist",
        "bip01_r_thightwist": "rig:RightUpLegTwist",
    }
    for raw, expected in twist_cases.items():
        assert canonicalize_bone_name(raw) == expected
        assert is_humanoid_bone(raw) is True
        assert is_core_bone(raw) is False


def test_canonical_61_bones_preserved() -> None:
    """All 61 canonical rig:* bones (25 core + 30 finger + 6 twist) must be preserved verbatim."""
    assert len(CORE_BONE_SUFFIXES) == 25
    assert len(FINGER_BONE_SUFFIXES) == 30
    assert len(TWIST_BONE_SUFFIXES) == 6
    assert len(ALL_HUMANOID_BONE_SUFFIXES) == 61

    for suffix in ALL_HUMANOID_BONE_SUFFIXES:
        canonical = f"rig:{suffix}"
        assert canonicalize_bone_name(canonical) == canonical
        assert normalize_rig_prefix(canonical) == canonical
        assert is_humanoid_bone(canonical) is True
        assert bone_suffix(canonical) == suffix

    for suffix in CORE_BONE_SUFFIXES:
        assert is_core_bone(f"rig:{suffix}") is True

    for suffix in FINGER_BONE_SUFFIXES:
        assert is_core_bone(f"rig:{suffix}") is False

    for suffix in TWIST_BONE_SUFFIXES:
        assert is_core_bone(f"rig:{suffix}") is False


def test_backward_compatibility_with_25_bone_rigs() -> None:
    """RigMapping.controlled on a 25-bone rig must return strictly the 22 active core bones."""
    from v2bm.rig.kinematics import HumanoidRig, RigBone
    import numpy as np

    # Build a mock 25-bone rig (all core bones)
    bones = {}
    for name in CORE_BONE_SUFFIXES:
        canon = f"rig:{name}"
        bones[canon] = RigBone(
            name=canon,
            parent=None,
            children=[],
            pre_rotation=np.eye(3),
            lcl_translation_cm=np.zeros(3),
            rest_world=np.eye(4),
            rest_local=np.eye(4),
            pre_rotation_euler_deg=np.zeros(3),
        )

    mock_rig = HumanoidRig(
        bones=bones,
        order=[f"rig:{s}" for s in CORE_BONE_SUFFIXES],
        meters_per_unit=0.01,
        source=Path("mock.fbx"),
    )

    mapping = RigMapping(
        skeleton_rig=mock_rig,
        target_rig=mock_rig,
        basis_change={},
        translation_scale=1.0,
        r_world_align=np.eye(3),
        max_basis_deviation_deg=0.0,
    )

    # Must match CORE_CONTROLLED_BONES exactly (22 bones)
    assert mapping.controlled == CORE_CONTROLLED_BONES
    assert len(mapping.controlled) == 22
    for finger in FINGER_BONE_SUFFIXES:
        assert f"rig:{finger}" not in mapping.controlled
    for twist in TWIST_BONE_SUFFIXES:
        assert f"rig:{twist}" not in mapping.controlled


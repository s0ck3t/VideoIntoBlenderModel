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
    CORE_BONE_SUFFIXES,
    HUMANOID_BONE_ALIASES,
    bone_suffix,
    canonicalize_bone_name,
    is_core_bone,
    normalize_rig_prefix,
)

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
    """Non-core bones (e.g. finger bones, props, accessories) are not incorrectly mapped."""
    assert canonicalize_bone_name("mixamorig:LeftHandThumb1") == "LeftHandThumb1"
    assert is_core_bone("mixamorig:LeftHandThumb1") is False
    assert is_core_bone("Camera") is False
    assert is_core_bone("Sword_Prop") is False


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


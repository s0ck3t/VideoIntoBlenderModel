"""Parse a binary Humanoid FBX and extract the core skeleton.

Reads the binary FBX directly (PreRotation is a Maya/FBX property, so Assimp
cannot be used) and returns the core-bone skeleton as a plain dict — the same
structure `kinematics.rig_from_skeleton_data` consumes.
"""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# Humanoid body chain only: no finger bones. End sites keep bone directions.
CORE_BONE_SUFFIXES = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "HeadTop_End",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "LeftToe_End",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "RightToe_End",
)

# Body24-24 (skeleton_motion.npz) → Humanoid Humanoid Rig.
HUMAN_TO_HUMANOID = {
    "pelvis": "rig:Hips",
    "left_hip": "rig:LeftUpLeg",
    "right_hip": "rig:RightUpLeg",
    "spine1": "rig:Spine",
    "left_knee": "rig:LeftLeg",
    "right_knee": "rig:RightLeg",
    "spine2": "rig:Spine1",
    "left_ankle": "rig:LeftFoot",
    "right_ankle": "rig:RightFoot",
    "spine3": "rig:Spine2",
    "left_foot": "rig:LeftToeBase",
    "right_foot": "rig:RightToeBase",
    "neck": "rig:Neck",
    "left_collar": "rig:LeftShoulder",
    "right_collar": "rig:RightShoulder",
    "head": "rig:Head",
    "left_shoulder": "rig:LeftArm",
    "right_shoulder": "rig:RightArm",
    "left_elbow": "rig:LeftForeArm",
    "right_elbow": "rig:RightForeArm",
    "left_wrist": "rig:LeftHand",
    "right_wrist": "rig:RightHand",
    "left_hand": "rig:LeftHand",
    "right_hand": "rig:RightHand",
}

# Humanoid bones that have no 1:1 Body24-24 joint (chain tips / extra spine).
HUMANOID_WITHOUT_HUMAN = (
    "rig:HeadTop_End",
    "rig:LeftToe_End",
    "rig:RightToe_End",
)

# Stage 1 prefix stripping: namespaces commonly appended by DCC tools and game engines.
_PREFIX_STRIP_RE = re.compile(
    r"^(?:rig\d*:|mixamorig\d*:|armature[:|_|]|bip\d+_(?:[clr]_)?|bip\d+_|j_bip_|character_)",
    re.IGNORECASE,
)
_RIG_PREFIX_RE = re.compile(r"^rig\d*:", re.IGNORECASE)

# Stage 2: Comprehensive case-insensitive humanoid bone alias dictionary.
# Maps standard third-party humanoid bone conventions (Mixamo, bare names,
# Quaternius, Kenney, MakeHuman/MPFB2, VRM/Unity, 3ds Max Biped) to canonical rig:* bones.
HUMANOID_BONE_ALIASES: dict[str, str] = {
    # Hips / Pelvis / Root
    "hips": "rig:Hips",
    "hip": "rig:Hips",
    "pelvis": "rig:Hips",
    "root": "rig:Hips",
    "c_hips": "rig:Hips",
    "j_bip_c_hips": "rig:Hips",
    "bip01_pelvis": "rig:Hips",
    "bip_pelvis": "rig:Hips",
    "pelvis_root": "rig:Hips",
    "hip_root": "rig:Hips",

    # Spine chain
    "spine": "rig:Spine",
    "spine_01": "rig:Spine",
    "spine_1": "rig:Spine",
    "spine01": "rig:Spine",
    "spine0": "rig:Spine",
    "spine_0": "rig:Spine",
    "lower_spine": "rig:Spine",
    "abdomen": "rig:Spine",
    "c_spine": "rig:Spine",
    "j_bip_c_spine": "rig:Spine",
    "bip01_spine": "rig:Spine",
    "bip_spine": "rig:Spine",

    # Spine1 / Chest
    "spine1": "rig:Spine1",
    "spine_02": "rig:Spine1",
    "spine_2": "rig:Spine1",
    "spine02": "rig:Spine1",
    "chest": "rig:Spine1",
    "c_chest": "rig:Spine1",
    "j_bip_c_chest": "rig:Spine1",
    "bip01_spine1": "rig:Spine1",
    "bip_spine1": "rig:Spine1",
    "middle_spine": "rig:Spine1",

    # Spine2 / UpperChest
    "spine2": "rig:Spine2",
    "spine_03": "rig:Spine2",
    "spine_3": "rig:Spine2",
    "spine03": "rig:Spine2",
    "upperchest": "rig:Spine2",
    "upper_chest": "rig:Spine2",
    "c_upperchest": "rig:Spine2",
    "j_bip_c_upperchest": "rig:Spine2",
    "bip01_spine2": "rig:Spine2",
    "bip_spine2": "rig:Spine2",
    "upper_spine": "rig:Spine2",

    # Neck
    "neck": "rig:Neck",
    "neck_01": "rig:Neck",
    "neck_1": "rig:Neck",
    "neck01": "rig:Neck",
    "neck1": "rig:Neck",
    "c_neck": "rig:Neck",
    "j_bip_c_neck": "rig:Neck",
    "bip01_neck": "rig:Neck",
    "bip_neck": "rig:Neck",
    "lower_neck": "rig:Neck",

    # Head
    "head": "rig:Head",
    "head_01": "rig:Head",
    "head_1": "rig:Head",
    "head01": "rig:Head",
    "head1": "rig:Head",
    "c_head": "rig:Head",
    "j_bip_c_head": "rig:Head",
    "bip01_head": "rig:Head",
    "bip_head": "rig:Head",

    # HeadTop_End
    "headtop_end": "rig:HeadTop_End",
    "head_end": "rig:HeadTop_End",
    "head_top": "rig:HeadTop_End",
    "headtop": "rig:HeadTop_End",
    "headtopend": "rig:HeadTop_End",
    "head_nub": "rig:HeadTop_End",
    "c_headtop_end": "rig:HeadTop_End",
    "bip01_headnub": "rig:HeadTop_End",
    "bip01_head_end": "rig:HeadTop_End",

    # Left Shoulder / Clavicle
    "leftshoulder": "rig:LeftShoulder",
    "left_shoulder": "rig:LeftShoulder",
    "shoulder.l": "rig:LeftShoulder",
    "shoulder_l": "rig:LeftShoulder",
    "l_shoulder": "rig:LeftShoulder",
    "clavicle.l": "rig:LeftShoulder",
    "clavicle_l": "rig:LeftShoulder",
    "l_clavicle": "rig:LeftShoulder",
    "leftclavicle": "rig:LeftShoulder",
    "left_clavicle": "rig:LeftShoulder",
    "collar.l": "rig:LeftShoulder",
    "collar_l": "rig:LeftShoulder",
    "l_collar": "rig:LeftShoulder",
    "leftcollar": "rig:LeftShoulder",
    "left_collar": "rig:LeftShoulder",
    "bip01_l_clavicle": "rig:LeftShoulder",
    "j_bip_l_shoulder": "rig:LeftShoulder",

    # Left Arm / Upper Arm
    "leftarm": "rig:LeftArm",
    "left_arm": "rig:LeftArm",
    "leftupperarm": "rig:LeftArm",
    "left_upperarm": "rig:LeftArm",
    "left_upper_arm": "rig:LeftArm",
    "upperarm.l": "rig:LeftArm",
    "upperarm_l": "rig:LeftArm",
    "upper_arm.l": "rig:LeftArm",
    "upper_arm_l": "rig:LeftArm",
    "l_upperarm": "rig:LeftArm",
    "l_upper_arm": "rig:LeftArm",
    "l_arm": "rig:LeftArm",
    "arm.l": "rig:LeftArm",
    "arm_l": "rig:LeftArm",
    "bip01_l_upperarm": "rig:LeftArm",
    "j_bip_l_upperarm": "rig:LeftArm",

    # Left Forearm / Lower Arm
    "leftforearm": "rig:LeftForeArm",
    "left_forearm": "rig:LeftForeArm",
    "leftlowerarm": "rig:LeftForeArm",
    "left_lowerarm": "rig:LeftForeArm",
    "left_lower_arm": "rig:LeftForeArm",
    "lowerarm.l": "rig:LeftForeArm",
    "lowerarm_l": "rig:LeftForeArm",
    "lower_arm.l": "rig:LeftForeArm",
    "lower_arm_l": "rig:LeftForeArm",
    "forearm.l": "rig:LeftForeArm",
    "forearm_l": "rig:LeftForeArm",
    "l_forearm": "rig:LeftForeArm",
    "l_lowerarm": "rig:LeftForeArm",
    "l_lower_arm": "rig:LeftForeArm",
    "bip01_l_forearm": "rig:LeftForeArm",
    "j_bip_l_lowerarm": "rig:LeftForeArm",

    # Left Hand / Wrist
    "lefthand": "rig:LeftHand",
    "left_hand": "rig:LeftHand",
    "hand.l": "rig:LeftHand",
    "hand_l": "rig:LeftHand",
    "l_hand": "rig:LeftHand",
    "wrist.l": "rig:LeftHand",
    "wrist_l": "rig:LeftHand",
    "l_wrist": "rig:LeftHand",
    "leftwrist": "rig:LeftHand",
    "left_wrist": "rig:LeftHand",
    "bip01_l_hand": "rig:LeftHand",
    "j_bip_l_hand": "rig:LeftHand",

    # Right Shoulder / Clavicle
    "rightshoulder": "rig:RightShoulder",
    "right_shoulder": "rig:RightShoulder",
    "shoulder.r": "rig:RightShoulder",
    "shoulder_r": "rig:RightShoulder",
    "r_shoulder": "rig:RightShoulder",
    "clavicle.r": "rig:RightShoulder",
    "clavicle_r": "rig:RightShoulder",
    "r_clavicle": "rig:RightShoulder",
    "rightclavicle": "rig:RightShoulder",
    "right_clavicle": "rig:RightShoulder",
    "collar.r": "rig:RightShoulder",
    "collar_r": "rig:RightShoulder",
    "r_collar": "rig:RightShoulder",
    "rightcollar": "rig:RightShoulder",
    "right_collar": "rig:RightShoulder",
    "bip01_r_clavicle": "rig:RightShoulder",
    "j_bip_r_shoulder": "rig:RightShoulder",

    # Right Arm / Upper Arm
    "rightarm": "rig:RightArm",
    "right_arm": "rig:RightArm",
    "rightupperarm": "rig:RightArm",
    "right_upperarm": "rig:RightArm",
    "right_upper_arm": "rig:RightArm",
    "upperarm.r": "rig:RightArm",
    "upperarm_r": "rig:RightArm",
    "upper_arm.r": "rig:RightArm",
    "upper_arm_r": "rig:RightArm",
    "r_upperarm": "rig:RightArm",
    "r_upper_arm": "rig:RightArm",
    "r_arm": "rig:RightArm",
    "arm.r": "rig:RightArm",
    "arm_r": "rig:RightArm",
    "bip01_r_upperarm": "rig:RightArm",
    "j_bip_r_upperarm": "rig:RightArm",

    # Right Forearm / Lower Arm
    "rightforearm": "rig:RightForeArm",
    "right_forearm": "rig:RightForeArm",
    "rightlowerarm": "rig:RightForeArm",
    "right_lowerarm": "rig:RightForeArm",
    "right_lower_arm": "rig:RightForeArm",
    "lowerarm.r": "rig:RightForeArm",
    "lowerarm_r": "rig:RightForeArm",
    "lower_arm.r": "rig:RightForeArm",
    "lower_arm_r": "rig:RightForeArm",
    "forearm.r": "rig:RightForeArm",
    "forearm_r": "rig:RightForeArm",
    "r_forearm": "rig:RightForeArm",
    "r_lowerarm": "rig:RightForeArm",
    "r_lower_arm": "rig:RightForeArm",
    "bip01_r_forearm": "rig:RightForeArm",
    "j_bip_r_lowerarm": "rig:RightForeArm",

    # Right Hand / Wrist
    "righthand": "rig:RightHand",
    "right_hand": "rig:RightHand",
    "hand.r": "rig:RightHand",
    "hand_r": "rig:RightHand",
    "r_hand": "rig:RightHand",
    "wrist.r": "rig:RightHand",
    "wrist_r": "rig:RightHand",
    "r_wrist": "rig:RightHand",
    "rightwrist": "rig:RightHand",
    "right_wrist": "rig:RightHand",
    "bip01_r_hand": "rig:RightHand",
    "j_bip_r_hand": "rig:RightHand",

    # Left UpLeg / Thigh
    "leftupleg": "rig:LeftUpLeg",
    "left_upleg": "rig:LeftUpLeg",
    "leftupperleg": "rig:LeftUpLeg",
    "left_upperleg": "rig:LeftUpLeg",
    "left_upper_leg": "rig:LeftUpLeg",
    "upperleg.l": "rig:LeftUpLeg",
    "upperleg_l": "rig:LeftUpLeg",
    "upper_leg.l": "rig:LeftUpLeg",
    "upper_leg_l": "rig:LeftUpLeg",
    "thigh.l": "rig:LeftUpLeg",
    "thigh_l": "rig:LeftUpLeg",
    "l_thigh": "rig:LeftUpLeg",
    "l_upleg": "rig:LeftUpLeg",
    "l_upperleg": "rig:LeftUpLeg",
    "leftthigh": "rig:LeftUpLeg",
    "left_thigh": "rig:LeftUpLeg",
    "bip01_l_thigh": "rig:LeftUpLeg",
    "j_bip_l_upperleg": "rig:LeftUpLeg",

    # Left Leg / Calf / Shin
    "leftleg": "rig:LeftLeg",
    "left_leg": "rig:LeftLeg",
    "leftlowerleg": "rig:LeftLeg",
    "left_lowerleg": "rig:LeftLeg",
    "left_lower_leg": "rig:LeftLeg",
    "lowerleg.l": "rig:LeftLeg",
    "lowerleg_l": "rig:LeftLeg",
    "lower_leg.l": "rig:LeftLeg",
    "lower_leg_l": "rig:LeftLeg",
    "calf.l": "rig:LeftLeg",
    "calf_l": "rig:LeftLeg",
    "l_calf": "rig:LeftLeg",
    "l_leg": "rig:LeftLeg",
    "l_lowerleg": "rig:LeftLeg",
    "shin.l": "rig:LeftLeg",
    "shin_l": "rig:LeftLeg",
    "l_shin": "rig:LeftLeg",
    "leftcalf": "rig:LeftLeg",
    "left_calf": "rig:LeftLeg",
    "leftshin": "rig:LeftLeg",
    "left_shin": "rig:LeftLeg",
    "bip01_l_calf": "rig:LeftLeg",
    "j_bip_l_lowerleg": "rig:LeftLeg",

    # Left Foot / Ankle
    "leftfoot": "rig:LeftFoot",
    "left_foot": "rig:LeftFoot",
    "foot.l": "rig:LeftFoot",
    "foot_l": "rig:LeftFoot",
    "l_foot": "rig:LeftFoot",
    "ankle.l": "rig:LeftFoot",
    "ankle_l": "rig:LeftFoot",
    "l_ankle": "rig:LeftFoot",
    "leftankle": "rig:LeftFoot",
    "left_ankle": "rig:LeftFoot",
    "bip01_l_foot": "rig:LeftFoot",
    "j_bip_l_foot": "rig:LeftFoot",

    # Left ToeBase / Ball / Toes
    "lefttoebase": "rig:LeftToeBase",
    "left_toebase": "rig:LeftToeBase",
    "lefttoe_base": "rig:LeftToeBase",
    "toebase.l": "rig:LeftToeBase",
    "toebase_l": "rig:LeftToeBase",
    "l_toebase": "rig:LeftToeBase",
    "lefttoes": "rig:LeftToeBase",
    "left_toes": "rig:LeftToeBase",
    "toes.l": "rig:LeftToeBase",
    "toes_l": "rig:LeftToeBase",
    "l_toes": "rig:LeftToeBase",
    "toe.l": "rig:LeftToeBase",
    "toe_l": "rig:LeftToeBase",
    "l_toe": "rig:LeftToeBase",
    "ball.l": "rig:LeftToeBase",
    "ball_l": "rig:LeftToeBase",
    "l_ball": "rig:LeftToeBase",
    "leftball": "rig:LeftToeBase",
    "left_ball": "rig:LeftToeBase",
    "toe0.l": "rig:LeftToeBase",
    "toe0_l": "rig:LeftToeBase",
    "l_toe0": "rig:LeftToeBase",
    "bip01_l_toe0": "rig:LeftToeBase",
    "j_bip_l_toebase": "rig:LeftToeBase",
    "j_bip_l_toes": "rig:LeftToeBase",

    # Left Toe_End
    "lefttoe_end": "rig:LeftToe_End",
    "left_toe_end": "rig:LeftToe_End",
    "lefttoes_end": "rig:LeftToe_End",
    "left_toes_end": "rig:LeftToe_End",
    "toe_end.l": "rig:LeftToe_End",
    "toe_end_l": "rig:LeftToe_End",
    "toes_end.l": "rig:LeftToe_End",
    "toes_end_l": "rig:LeftToe_End",
    "l_toe_end": "rig:LeftToe_End",
    "l_toes_end": "rig:LeftToe_End",
    "toeend.l": "rig:LeftToe_End",
    "toeend_l": "rig:LeftToe_End",
    "bip01_l_toe0nub": "rig:LeftToe_End",
    "bip01_l_toeend": "rig:LeftToe_End",

    # Right UpLeg / Thigh
    "rightupleg": "rig:RightUpLeg",
    "right_upleg": "rig:RightUpLeg",
    "rightupperleg": "rig:RightUpLeg",
    "right_upperleg": "rig:RightUpLeg",
    "right_upper_leg": "rig:RightUpLeg",
    "upperleg.r": "rig:RightUpLeg",
    "upperleg_r": "rig:RightUpLeg",
    "upper_leg.r": "rig:RightUpLeg",
    "upper_leg_r": "rig:RightUpLeg",
    "thigh.r": "rig:RightUpLeg",
    "thigh_r": "rig:RightUpLeg",
    "r_thigh": "rig:RightUpLeg",
    "r_upleg": "rig:RightUpLeg",
    "r_upperleg": "rig:RightUpLeg",
    "rightthigh": "rig:RightUpLeg",
    "right_thigh": "rig:RightUpLeg",
    "bip01_r_thigh": "rig:RightUpLeg",
    "j_bip_r_upperleg": "rig:RightUpLeg",

    # Right Leg / Calf / Shin
    "rightleg": "rig:RightLeg",
    "right_leg": "rig:RightLeg",
    "rightlowerleg": "rig:RightLeg",
    "right_lowerleg": "rig:RightLeg",
    "right_lower_leg": "rig:RightLeg",
    "lowerleg.r": "rig:RightLeg",
    "lowerleg_r": "rig:RightLeg",
    "lower_leg.r": "rig:RightLeg",
    "lower_leg_r": "rig:RightLeg",
    "calf.r": "rig:RightLeg",
    "calf_r": "rig:RightLeg",
    "r_calf": "rig:RightLeg",
    "r_leg": "rig:RightLeg",
    "r_lowerleg": "rig:RightLeg",
    "shin.r": "rig:RightLeg",
    "shin_r": "rig:RightLeg",
    "r_shin": "rig:RightLeg",
    "rightcalf": "rig:RightLeg",
    "right_calf": "rig:RightLeg",
    "rightshin": "rig:RightLeg",
    "right_shin": "rig:RightLeg",
    "bip01_r_calf": "rig:RightLeg",
    "j_bip_r_lowerleg": "rig:RightLeg",

    # Right Foot / Ankle
    "rightfoot": "rig:RightFoot",
    "right_foot": "rig:RightFoot",
    "foot.r": "rig:RightFoot",
    "foot_r": "rig:RightFoot",
    "r_foot": "rig:RightFoot",
    "ankle.r": "rig:RightFoot",
    "ankle_r": "rig:RightFoot",
    "r_ankle": "rig:RightFoot",
    "rightankle": "rig:RightFoot",
    "right_ankle": "rig:RightFoot",
    "bip01_r_foot": "rig:RightFoot",
    "j_bip_r_foot": "rig:RightFoot",

    # Right ToeBase / Ball / Toes
    "righttoebase": "rig:RightToeBase",
    "right_toebase": "rig:RightToeBase",
    "righttoe_base": "rig:RightToeBase",
    "toebase.r": "rig:RightToeBase",
    "toebase_r": "rig:RightToeBase",
    "r_toebase": "rig:RightToeBase",
    "righttoes": "rig:RightToeBase",
    "right_toes": "rig:RightToeBase",
    "toes.r": "rig:RightToeBase",
    "toes_r": "rig:RightToeBase",
    "r_toes": "rig:RightToeBase",
    "toe.r": "rig:RightToeBase",
    "toe_r": "rig:RightToeBase",
    "r_toe": "rig:RightToeBase",
    "ball.r": "rig:RightToeBase",
    "ball_r": "rig:RightToeBase",
    "r_ball": "rig:RightToeBase",
    "rightball": "rig:RightToeBase",
    "right_ball": "rig:RightToeBase",
    "toe0.r": "rig:RightToeBase",
    "toe0_r": "rig:RightToeBase",
    "r_toe0": "rig:RightToeBase",
    "bip01_r_toe0": "rig:RightToeBase",
    "j_bip_r_toebase": "rig:RightToeBase",
    "j_bip_r_toes": "rig:RightToeBase",

    # Right Toe_End
    "righttoe_end": "rig:RightToe_End",
    "right_toe_end": "rig:RightToe_End",
    "righttoes_end": "rig:RightToe_End",
    "right_toes_end": "rig:RightToe_End",
    "toe_end.r": "rig:RightToe_End",
    "toe_end_r": "rig:RightToe_End",
    "toes_end.r": "rig:RightToe_End",
    "toes_end_r": "rig:RightToe_End",
    "r_toe_end": "rig:RightToe_End",
    "r_toes_end": "rig:RightToe_End",
    "toeend.r": "rig:RightToe_End",
    "toeend_r": "rig:RightToe_End",
    "bip01_r_toe0nub": "rig:RightToe_End",
    "bip01_r_toeend": "rig:RightToe_End",
}

FBX_HEADER = b"Kaydara FBX Binary  \x00\x1a\x00"
ARRAY_DTYPE = {
    b"f": ("<f", 4),
    b"d": ("<d", 8),
    b"l": ("<q", 8),
    b"i": ("<i", 4),
    b"b": ("<?", 1),
}


class FbxError(ValueError):
    pass


def _read(buf: memoryview, offset: int, fmt: str) -> tuple[Any, int]:
    size = struct.calcsize(fmt)
    return struct.unpack_from(fmt, buf, offset)[0], offset + size


def _read_array(buf: memoryview, offset: int, type_code: bytes) -> tuple[list[Any], int]:
    length, offset = _read(buf, offset, "<I")
    encoding, offset = _read(buf, offset, "<I")
    compressed_len, offset = _read(buf, offset, "<I")
    dtype, item_size = ARRAY_DTYPE[type_code]
    raw = bytes(buf[offset : offset + compressed_len])
    offset += compressed_len
    if encoding == 1:
        raw = zlib.decompress(raw)
    elif encoding != 0:
        raise FbxError(f"unsupported FBX array encoding {encoding}")
    expected = length * item_size
    if len(raw) != expected:
        raise FbxError(f"array size mismatch: {len(raw)} != {expected}")
    values = list(struct.unpack_from(f"<{length}{dtype[-1]}", raw, 0)) if length else []
    return values, offset


def _read_property(buf: memoryview, offset: int) -> tuple[Any, int]:
    type_code = bytes(buf[offset : offset + 1])
    offset += 1
    if type_code == b"Y":
        return _read(buf, offset, "<h")
    if type_code == b"C":
        value, offset = _read(buf, offset, "<B")
        return bool(value), offset
    if type_code == b"I":
        return _read(buf, offset, "<i")
    if type_code == b"F":
        return _read(buf, offset, "<f")
    if type_code == b"D":
        return _read(buf, offset, "<d")
    if type_code == b"L":
        return _read(buf, offset, "<q")
    if type_code in ARRAY_DTYPE:
        return _read_array(buf, offset, type_code)
    if type_code in (b"S", b"R"):
        length, offset = _read(buf, offset, "<I")
        data = bytes(buf[offset : offset + length])
        offset += length
        if type_code == b"S":
            return data.split(b"\x00")[0].decode("utf-8", errors="replace"), offset
        return data, offset
    raise FbxError(f"unknown FBX property type {type_code!r}")


def _is_null_record(buf: memoryview, offset: int, version: int) -> bool:
    size = 25 if version >= 7500 else 13
    return offset + size <= len(buf) and all(b == 0 for b in buf[offset : offset + size])


def _read_node(buf: memoryview, offset: int, version: int) -> tuple[dict[str, Any] | None, int]:
    start = offset
    if version >= 7500:
        end_offset, offset = _read(buf, offset, "<Q")
        n_props, offset = _read(buf, offset, "<Q")
        _prop_len, offset = _read(buf, offset, "<Q")
    else:
        end_offset, offset = _read(buf, offset, "<I")
        n_props, offset = _read(buf, offset, "<I")
        _prop_len, offset = _read(buf, offset, "<I")
    if end_offset == 0:
        return None, start + (25 if version >= 7500 else 13)
    name_len = buf[offset]
    offset += 1
    name = bytes(buf[offset : offset + name_len]).decode("ascii", errors="replace")
    offset += name_len
    props: list[Any] = []
    for _ in range(n_props):
        value, offset = _read_property(buf, offset)
        props.append(value)
    children: list[dict[str, Any]] = []
    while offset < end_offset:
        if _is_null_record(buf, offset, version):
            offset = end_offset
            break
        child, offset = _read_node(buf, offset, version)
        if child is None:
            break
        children.append(child)
    return {"name": name, "props": props, "children": children}, end_offset


def parse_fbx(path: Path) -> tuple[int, list[dict[str, Any]]]:
    data = path.read_bytes()
    if not data.startswith(FBX_HEADER):
        raise FbxError(f"{path} is not a binary FBX file")
    version = struct.unpack_from("<I", data, 23)[0]
    buf = memoryview(data)
    offset = 27
    nodes: list[dict[str, Any]] = []
    while offset < len(buf):
        if _is_null_record(buf, offset, version):
            break
        node, offset = _read_node(buf, offset, version)
        if node is None:
            break
        nodes.append(node)
    return version, nodes


def _children_named(node: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [c for c in node["children"] if c["name"] == name]


def _child_named(node: dict[str, Any], name: str) -> dict[str, Any] | None:
    for child in node["children"]:
        if child["name"] == name:
            return child
    return None


def _vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        out = np.array(default, dtype=np.float64)
        out[: arr.size] = arr
        return out
    return arr[:3].copy()


def parse_properties70(node: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    block = _child_named(node, "Properties70")
    if block is None:
        return props
    for child in block["children"]:
        if child["name"] != "P" or not child["props"]:
            continue
        raw = child["props"]
        key = str(raw[0])
        values = raw[4:] if len(raw) > 4 else []
        if len(values) == 1:
            props[key] = values[0]
        elif len(values) >= 3 and all(isinstance(v, (int, float)) for v in values[:3]):
            props[key] = [float(v) for v in values[:3]]
        else:
            props[key] = values
    return props


def euler_xyz_deg_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """FBX/Maya Euler XYZ: R = Rz @ Ry @ Rx (column vectors)."""
    x, y, z = np.deg2rad(np.asarray(euler_deg, dtype=np.float64)[:3])
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_quat_wxyz(rmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rmat, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if q[0] < 0:
        q = -q
    return q / n


def make_transform(
    translation: np.ndarray,
    pre_rot_deg: np.ndarray,
    lcl_rot_deg: np.ndarray,
    post_rot_deg: np.ndarray,
    scale: np.ndarray,
    rotation_offset: np.ndarray | None = None,
    rotation_pivot: np.ndarray | None = None,
) -> np.ndarray:
    """Maya/FBX local TRS with PreRotation (pivots/offsets identity on Humanoid).

    M = T * Roff * Rp * Rpre * R * Rpost^{-1} * Rp^{-1} * S
    """
    t = _vec3(translation)
    s = _vec3(scale, (1.0, 1.0, 1.0))
    roff = _vec3(rotation_offset)
    rp = _vec3(rotation_pivot)
    rpre = euler_xyz_deg_to_matrix(pre_rot_deg)
    r = euler_xyz_deg_to_matrix(lcl_rot_deg)
    rpost = euler_xyz_deg_to_matrix(post_rot_deg)
    rot = rpre @ r @ rpost.T

    def translate(vec: np.ndarray) -> np.ndarray:
        m = np.eye(4)
        m[:3, 3] = vec
        return m

    t_mat = translate(t)
    roff_mat = translate(roff)
    rp_mat = translate(rp)
    rp_inv = translate(-rp)
    r_mat = np.eye(4)
    r_mat[:3, :3] = rot
    s_mat = np.diag([float(s[0]), float(s[1]), float(s[2]), 1.0])
    return t_mat @ roff_mat @ rp_mat @ r_mat @ rp_inv @ s_mat


def _tolist(arr: np.ndarray, ndigits: int = 12) -> list[Any]:
    def _round(x: float) -> float:
        v = round(float(x), ndigits)
        return 0.0 if abs(v) < 10 ** (-ndigits) else v

    if arr.ndim == 1:
        return [_round(x) for x in arr]
    return [[_round(x) for x in row] for row in arr]


def _describe_path(path: Path) -> str:
    return str(path)


def canonicalize_bone_name(name: str) -> str:
    """Normalise a bone name from any supported humanoid convention to canonical `rig:*`.

    Two-stage normalisation:
    - Stage 1: Strip namespace prefixes (`rig\\d*:`, `mixamorig\\d*:`, `Bip\\d+_`, `J_Bip_[CLR]_`, etc.).
    - Stage 2: Comprehensive case-insensitive alias dictionary mapping all standard third-party
      humanoid bone names (Mixamo, bare names, Quaternius, Kenney, MakeHuman/MPFB2, VRM)
      to canonical `rig:*` core bones.

    Guarantees 100% backwards compatibility by preserving exact `rig:<Suffix>` bones.
    """
    if not name:
        return ""
    name_str = str(name).strip()
    if not name_str:
        return ""

    # Preserve exact canonical rig:* bones immediately (100% backwards compatibility)
    if name_str.startswith("rig:"):
        suffix = name_str[4:]
        if suffix in CORE_BONE_SUFFIXES:
            return name_str

    # Direct check against alias table (covers fully-qualified names like J_Bip_L_UpperArm, Bip01_Pelvis, etc.)
    name_lower = name_str.lower()
    if name_lower in HUMANOID_BONE_ALIASES:
        return HUMANOID_BONE_ALIASES[name_lower]

    # Stage 1: Strip namespace / prefix
    token = _PREFIX_STRIP_RE.sub("", name_str, count=1)
    token_lower = token.lower()

    # Stage 2: Direct match against core bone suffixes (case-insensitive)
    for core_suffix in CORE_BONE_SUFFIXES:
        if token_lower == core_suffix.lower():
            return f"rig:{core_suffix}"

    # Stage 2: Match stripped token in alias dictionary
    if token_lower in HUMANOID_BONE_ALIASES:
        return HUMANOID_BONE_ALIASES[token_lower]

    # Handle dot/underscore variants: e.g. "upperarm.l" vs "upperarm_l" vs "upperarm l"
    token_norm = token_lower.replace(".", "_").replace(" ", "_").replace("-", "_")
    if token_norm in HUMANOID_BONE_ALIASES:
        return HUMANOID_BONE_ALIASES[token_norm]

    # Fallback: if input had rig\\d+: prefix (e.g. non-core bone like rig8:LeftHandThumb1), keep rig:
    if _RIG_PREFIX_RE.match(name_str):
        return _RIG_PREFIX_RE.sub("rig:", name_str, count=1)

    # Return stripped token for non-core limbs
    return token


def normalize_rig_prefix(name: str) -> str:
    """Normalise a bone name to canonical `rig:*`.

    Retained for full backwards compatibility; delegates to `canonicalize_bone_name`.
    """
    return canonicalize_bone_name(name)


def bone_suffix(name: str) -> str:
    """Extract the canonical suffix of a bone name (e.g. 'rig:Hips' -> 'Hips')."""
    canonical = canonicalize_bone_name(name)
    if canonical.startswith("rig:"):
        return canonical[4:]
    return name.split(":")[-1]


def is_core_bone(name: str) -> bool:
    """Check if a bone name corresponds to a canonical humanoid core bone."""
    canonical = canonicalize_bone_name(name)
    return bone_suffix(canonical) in CORE_BONE_SUFFIXES


def extract_objects(nodes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    objects: dict[int, dict[str, Any]] = {}
    for node in nodes:
        if node["name"] != "Objects":
            continue
        for child in node["children"]:
            if not child["props"]:
                continue
            uid = int(child["props"][0])
            name = normalize_rig_prefix(str(child["props"][1])) if len(child["props"]) > 1 else ""
            class_name = str(child["props"][2]) if len(child["props"]) > 2 else ""
            objects[uid] = {
                "uid": uid,
                "node_name": child["name"],
                "name": name,
                "class": class_name,
                "props": parse_properties70(child),
                "raw": child,
            }
    return objects


def extract_connections(nodes: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    links: list[tuple[str, int, int]] = []
    for node in nodes:
        if node["name"] != "Connections":
            continue
        for child in node["children"]:
            if child["name"] != "C" or len(child["props"]) < 3:
                continue
            kind = str(child["props"][0])
            src = int(child["props"][1])
            dst = int(child["props"][2])
            links.append((kind, src, dst))
    return links


def extract_global_settings(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    for node in nodes:
        if node["name"] == "GlobalSettings":
            return parse_properties70(node)
    return {}


def extract_bind_pose(objects: dict[int, dict[str, Any]]) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    for obj in objects.values():
        if obj["node_name"] != "Pose":
            continue
        for pose_node in _children_named(obj["raw"], "PoseNode"):
            node_elem = _child_named(pose_node, "Node")
            matrix_elem = _child_named(pose_node, "Matrix")
            if node_elem is None or matrix_elem is None or not matrix_elem["props"]:
                continue
            uid = int(node_elem["props"][0])
            values = matrix_elem["props"][0]
            mat = np.asarray(values, dtype=np.float64).reshape(4, 4).T
            poses[uid] = mat
    return poses


def collect_limb_nodes(objects: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    limbs: dict[int, dict[str, Any]] = {}
    for uid, obj in objects.items():
        if obj["node_name"] != "Model":
            continue
        if obj["class"] != "LimbNode":
            continue
        limbs[uid] = obj
    return limbs


def build_hierarchy(
    limbs: dict[int, dict[str, Any]],
    connections: list[tuple[str, int, int]],
) -> dict[int, int | None]:
    parent_of: dict[int, int | None] = {uid: None for uid in limbs}
    for kind, src, dst in connections:
        if kind != "OO":
            continue
        if src in limbs and dst in limbs:
            parent_of[src] = dst
    return parent_of


def bone_local_matrix(props: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    translation = _vec3(props.get("Lcl Translation"))
    lcl_rot = _vec3(props.get("Lcl Rotation"))
    pre_rot = _vec3(props.get("PreRotation"))
    post_rot = _vec3(props.get("PostRotation"))
    scale = _vec3(props.get("Lcl Scaling"), (1.0, 1.0, 1.0))
    rotation_offset = _vec3(props.get("RotationOffset"))
    rotation_pivot = _vec3(props.get("RotationPivot"))
    local = make_transform(
        translation,
        pre_rot,
        lcl_rot,
        post_rot,
        scale,
        rotation_offset=rotation_offset,
        rotation_pivot=rotation_pivot,
    )
    parts = {
        "lcl_translation": translation,
        "lcl_rotation_euler_deg": lcl_rot,
        "pre_rotation_euler_deg": pre_rot,
        "post_rotation_euler_deg": post_rot,
        "lcl_scaling": scale,
        "rotation_offset": rotation_offset,
        "rotation_pivot": rotation_pivot,
    }
    return local, parts


def compose_world(
    parent_of: dict[int, int | None],
    local_of: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    world: dict[int, np.ndarray] = {}

    def eval_uid(uid: int) -> np.ndarray:
        if uid in world:
            return world[uid]
        parent = parent_of.get(uid)
        local = local_of[uid]
        if parent is None:
            world[uid] = local.copy()
        else:
            world[uid] = eval_uid(parent) @ local
        return world[uid]

    for uid in local_of:
        eval_uid(uid)
    return world


CANONICAL_HUMANOID_SPAN_M = 1.70
MIN_VALID_HUMANOID_SPAN_M = 0.80
MAX_VALID_HUMANOID_SPAN_M = 2.80


def unit_to_meters(
    settings: dict[str, Any],
    world_positions: list[np.ndarray] | None = None,
) -> float:
    """Resolve effective metres-per-FBX-unit, auto-calibrating when FBX header units mismatch geometry.

    FBX defaults to centimetres (`0.01 * UnitScaleFactor`), but many third-party tools
    export metre-scale (`~1.75` units) or game-unit (`~3.76` units) skeletons while
    leaving `UnitScaleFactor == 1.0`. When `world_positions` of the core skeleton are
    supplied and the header scale yields an implausible humanoid span outside
    `[0.80 m, 2.80 m]`, the conversion factor is automatically calibrated from the
    intrinsic 3D bounding-box span of the rest skeleton.
    """
    scale = float(settings.get("UnitScaleFactor", 1.0) or 1.0)
    header_mpu = 0.01 * scale
    if not world_positions or len(world_positions) < 2:
        return header_mpu

    pts = np.asarray(world_positions, dtype=np.float64).reshape(-1, 3)
    raw_span = float(np.max(np.ptp(pts, axis=0)))
    if raw_span <= 1e-6:
        return header_mpu

    apparent_m = raw_span * header_mpu
    if MIN_VALID_HUMANOID_SPAN_M <= apparent_m <= MAX_VALID_HUMANOID_SPAN_M:
        return header_mpu

    return CANONICAL_HUMANOID_SPAN_M / raw_span


def axis_name(index: int, sign: int) -> str:
    names = {0: "X", 1: "Y", 2: "Z"}
    prefix = "+" if int(sign) >= 0 else "-"
    return f"{prefix}{names.get(int(index), '?')}"


def extract_skeleton(fbx_path: Path) -> dict[str, Any]:
    version, nodes = parse_fbx(fbx_path)
    objects = extract_objects(nodes)
    connections = extract_connections(nodes)
    settings = extract_global_settings(nodes)
    bind_pose = extract_bind_pose(objects)
    limbs = collect_limb_nodes(objects)
    parent_of = build_hierarchy(limbs, connections)

    local_of: dict[int, np.ndarray] = {}
    parts_of: dict[int, dict[str, np.ndarray]] = {}
    for uid, obj in limbs.items():
        local, parts = bone_local_matrix(obj["props"])
        local_of[uid] = local
        parts_of[uid] = parts
    world_of = compose_world(parent_of, local_of)

    core_uids = [uid for uid, obj in limbs.items() if is_core_bone(obj["name"])]
    children_of: dict[int, list[int]] = {uid: [] for uid in core_uids}
    for uid in core_uids:
        parent = parent_of[uid]
        if parent in children_of:
            children_of[parent].append(uid)

    def sort_children(uids: list[int]) -> list[int]:
        order = {name: i for i, name in enumerate(CORE_BONE_SUFFIXES)}
        return sorted(uids, key=lambda u: order.get(bone_suffix(limbs[u]["name"]), 999))

    roots = [uid for uid in core_uids if parent_of[uid] not in set(core_uids)]
    ordered: list[int] = []

    def walk(uid: int) -> None:
        ordered.append(uid)
        for child in sort_children(children_of[uid]):
            walk(child)

    for root in sort_children(roots):
        walk(root)

    core_world_pts = [world_of[uid][:3, 3] for uid in core_uids]
    meters_per_fbx = unit_to_meters(settings, core_world_pts)
    bones = []
    for uid in ordered:
        obj = limbs[uid]
        parts = parts_of[uid]
        local = local_of[uid]
        world = world_of[uid]
        parent_uid = parent_of[uid]
        parent_name = limbs[parent_uid]["name"] if parent_uid in limbs else None
        child_names = [limbs[c]["name"] for c in sort_children(children_of[uid])]
        bind = bind_pose.get(uid)
        bind_err = None
        if bind is not None:
            bind_err = float(np.linalg.norm(world - bind))
        pre = parts["pre_rotation_euler_deg"]
        lcl_rot = parts["lcl_rotation_euler_deg"]
        post_rot = parts["post_rotation_euler_deg"]
        rest_rot_matrix = (
            euler_xyz_deg_to_matrix(pre)
            @ euler_xyz_deg_to_matrix(lcl_rot)
            @ euler_xyz_deg_to_matrix(post_rot).T
        )
        bones.append(
            {
                "name": obj["name"],
                "parent": parent_name,
                "children": child_names,
                "pre_rotation_euler_deg": _tolist(pre),
                "pre_rotation_matrix": _tolist(rest_rot_matrix),
                "lcl_translation": _tolist(parts["lcl_translation"]),
                "lcl_rotation_euler_deg": _tolist(lcl_rot),
                "lcl_scaling": _tolist(parts["lcl_scaling"]),
                "post_rotation_euler_deg": _tolist(post_rot),
                "rest_local": {
                    "matrix": _tolist(local),
                    "translation": _tolist(local[:3, 3]),
                    "rotation_matrix": _tolist(local[:3, :3]),
                    "quaternion_wxyz": _tolist(matrix_to_quat_wxyz(local[:3, :3])),
                    "scale": _tolist(np.array([np.linalg.norm(local[:3, i]) for i in range(3)])),
                },
                "rest_world": {
                    "matrix": _tolist(world),
                    "translation": _tolist(world[:3, 3]),
                    "translation_m": _tolist(world[:3, 3] * meters_per_fbx),
                    "rotation_matrix": _tolist(world[:3, :3]),
                    "quaternion_wxyz": _tolist(matrix_to_quat_wxyz(world[:3, :3])),
                },
                "bind_pose_world_matrix": _tolist(bind) if bind is not None else None,
                "bind_pose_match_error": bind_err,
            }
        )

    hierarchy = [
        {
            "name": limbs[uid]["name"],
            "parent": limbs[parent_of[uid]]["name"] if parent_of[uid] in limbs else None,
            "children": [limbs[c]["name"] for c in sort_children(children_of[uid])],
        }
        for uid in ordered
    ]

    return {
        "source_fbx": _describe_path(fbx_path),
        "fbx_version": version,
        "character": fbx_path.stem,
        "coord_system": {
            "up": axis_name(int(settings.get("UpAxis", 1)), int(settings.get("UpAxisSign", 1))),
            "front": axis_name(int(settings.get("FrontAxis", 2)), int(settings.get("FrontAxisSign", 1))),
            "coord": axis_name(int(settings.get("CoordAxis", 0)), int(settings.get("CoordAxisSign", 1))),
            "handedness": "right",
            "rotation_order": "XYZ",
            "fbx_unit": "centimeter" if abs(float(settings.get("UnitScaleFactor", 1.0)) - 1.0) < 1e-6 else "custom",
            "unit_scale_factor": float(settings.get("UnitScaleFactor", 1.0)),
            "meters_per_fbx_unit": meters_per_fbx,
        },
        "transform_convention": (
            "Maya/FBX: M_local = T * Roff * Rp * Rpre * R * Rpost^{-1} * Rp^{-1} * S; "
            "M_world = M_parent_world * M_local. Euler XYZ in degrees, column-vector matrices."
        ),
        "root": limbs[roots[0]]["name"] if roots else None,
        "bone_count": len(bones),
        "hierarchy": hierarchy,
        "bones": bones,
    }

"""Procedural CC0 61-Bone Articulated Humanoid Character Rig Generator for Blender 5.2.2 LTS.

Generates assets/rigs/humanoid_articulated.fbx with:
- Exact 61-bone canonical hierarchy (25 core bones + 30 phalangeal finger bones + 6 limb twist bones).
- Uniform +Y edit bone tail orientation (head[1] + 5.0 cm) guaranteeing an identity rest rotation matrix
  (0 degrees) and eliminating FBX PreRotation / PostRotation anomalies.
- Dual skinned meshes:
  1. Alpha_Surface: Continuous organic humanoid manifold with 5 articulated digits per hand,
     voxel remeshed at 0.90 cm resolution to preserve digit separation without webbing,
     conformal UV unwrapped, and bound via analytical C1 Hermite smoothstep skin weights.
  2. Alpha_Joints: 12 articulated accent rings rigidly bound to host articulation bones,
     conformal UV unwrapped, and shaded smooth.
- Embedded 512x512 studio gradient PBR textures (slate-to-pearl for Alpha_Surface, dark titanium for Alpha_Joints).
- Strict partition of unity across all 61 bone weights (sum of weights == 1.0 with zero deviation).
- Headless execution with robust disk logging for Windows execution.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def log(msg: str, log_file: Path | None = None) -> None:
    """Write timestamped log entry to stdout and optional disk log file."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    formatted = f"[{stamp}] [GENERATOR] {msg}"
    print(formatted)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except OSError as exc:
            print(f"Warning: Failed to write to log file {log_file}: {exc}")


def smoothstep(edge0: float, edge1: float, x: float | np.ndarray) -> float | np.ndarray:
    """Hermite interpolation with zero first derivatives at edges."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_articulated_humanoid_fbx(output_fbx: Path, log_file: Path | None = None) -> bool:
    """Build and export continuous organic CC0 61-Bone Articulated Humanoid character asset.

    Parameters
    ----------
    output_fbx : Path
        Destination path for exported binary FBX asset.
    log_file : Path | None
        Destination path for detailed execution log.

    Returns
    -------
    bool
        True if generation and export succeeded, False otherwise.
    """
    import bpy

    output_fbx = Path(output_fbx).resolve()
    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    if log_file:
        log_file = Path(log_file).resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)

    log(f"Starting procedural CC0 61-bone articulated humanoid generation: {output_fbx}", log_file)

    # 1. Reset Blender workspace to clean factory state
    bpy.ops.wm.read_factory_settings(use_empty=True)
    log("Reset Blender workspace to empty factory scene", log_file)

    # 2. Construct Armature object
    arm_data = bpy.data.armatures.new("Armature")
    arm_obj = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    # 61 canonical bones specification: (name, parent, head_xyz_cm)
    # Uniform +Y edit bone tail offset (+5.0 cm) guarantees an identity (0 deg) rest rotation matrix
    bone_specs: list[tuple[str, str | None, tuple[float, float, float]]] = [
        # Spine chain (7 bones)
        ("rig:Hips", None, (0.0, 102.0, 0.0)),
        ("rig:Spine", "rig:Hips", (0.0, 112.0, 0.0)),
        ("rig:Spine1", "rig:Spine", (0.0, 124.0, 0.0)),
        ("rig:Spine2", "rig:Spine1", (0.0, 136.0, 0.0)),
        ("rig:Neck", "rig:Spine2", (0.0, 150.0, 0.0)),
        ("rig:Head", "rig:Neck", (0.0, 160.0, 0.0)),
        ("rig:HeadTop_End", "rig:Head", (0.0, 175.0, 0.0)),

        # Left Arm & Twists (5 bones)
        ("rig:LeftShoulder", "rig:Spine2", (5.0, 148.0, 0.0)),
        ("rig:LeftArm", "rig:LeftShoulder", (18.0, 148.0, 0.0)),
        ("rig:LeftArmTwist", "rig:LeftArm", (31.5, 148.0, 0.0)),
        ("rig:LeftForeArm", "rig:LeftArm", (45.0, 148.0, 0.0)),
        ("rig:LeftForeArmTwist", "rig:LeftForeArm", (58.5, 148.0, 0.0)),
        ("rig:LeftHand", "rig:LeftForeArm", (72.0, 148.0, 0.0)),

        # Left Hand 15 Phalangeal Fingers (Thumb, Index, Middle, Ring, Pinky)
        ("rig:LeftHandThumb1", "rig:LeftHand", (74.5, 147.2, 2.2)),
        ("rig:LeftHandThumb2", "rig:LeftHandThumb1", (77.0, 146.8, 3.5)),
        ("rig:LeftHandThumb3", "rig:LeftHandThumb2", (79.2, 146.4, 4.5)),

        ("rig:LeftHandIndex1", "rig:LeftHand", (76.8, 148.0, 1.8)),
        ("rig:LeftHandIndex2", "rig:LeftHandIndex1", (79.5, 148.0, 1.8)),
        ("rig:LeftHandIndex3", "rig:LeftHandIndex2", (82.0, 148.0, 1.8)),

        ("rig:LeftHandMiddle1", "rig:LeftHand", (77.0, 148.0, 0.6)),
        ("rig:LeftHandMiddle2", "rig:LeftHandMiddle1", (80.0, 148.0, 0.6)),
        ("rig:LeftHandMiddle3", "rig:LeftHandMiddle2", (82.8, 148.0, 0.6)),

        ("rig:LeftHandRing1", "rig:LeftHand", (76.6, 148.0, -0.6)),
        ("rig:LeftHandRing2", "rig:LeftHandRing1", (79.4, 148.0, -0.6)),
        ("rig:LeftHandRing3", "rig:LeftHandRing2", (81.8, 148.0, -0.6)),

        ("rig:LeftHandPinky1", "rig:LeftHand", (76.0, 148.0, -1.8)),
        ("rig:LeftHandPinky2", "rig:LeftHandPinky1", (78.5, 148.0, -1.8)),
        ("rig:LeftHandPinky3", "rig:LeftHandPinky2", (80.6, 148.0, -1.8)),

        # Right Arm & Twists (5 bones)
        ("rig:RightShoulder", "rig:Spine2", (-5.0, 148.0, 0.0)),
        ("rig:RightArm", "rig:RightShoulder", (-18.0, 148.0, 0.0)),
        ("rig:RightArmTwist", "rig:RightArm", (-31.5, 148.0, 0.0)),
        ("rig:RightForeArm", "rig:RightArm", (-45.0, 148.0, 0.0)),
        ("rig:RightForeArmTwist", "rig:RightForeArm", (-58.5, 148.0, 0.0)),
        ("rig:RightHand", "rig:RightForeArm", (-72.0, 148.0, 0.0)),

        # Right Hand 15 Phalangeal Fingers (Thumb, Index, Middle, Ring, Pinky)
        ("rig:RightHandThumb1", "rig:RightHand", (-74.5, 147.2, 2.2)),
        ("rig:RightHandThumb2", "rig:RightHandThumb1", (-77.0, 146.8, 3.5)),
        ("rig:RightHandThumb3", "rig:RightHandThumb2", (-79.2, 146.4, 4.5)),

        ("rig:RightHandIndex1", "rig:RightHand", (-76.8, 148.0, 1.8)),
        ("rig:RightHandIndex2", "rig:RightHandIndex1", (-79.5, 148.0, 1.8)),
        ("rig:RightHandIndex3", "rig:RightHandIndex2", (-82.0, 148.0, 1.8)),

        ("rig:RightHandMiddle1", "rig:RightHand", (-77.0, 148.0, 0.6)),
        ("rig:RightHandMiddle2", "rig:RightHandMiddle1", (-80.0, 148.0, 0.6)),
        ("rig:RightHandMiddle3", "rig:RightHandMiddle2", (-82.8, 148.0, 0.6)),

        ("rig:RightHandRing1", "rig:RightHand", (-76.6, 148.0, -0.6)),
        ("rig:RightHandRing2", "rig:RightHandRing1", (-79.4, 148.0, -0.6)),
        ("rig:RightHandRing3", "rig:RightHandRing2", (-81.8, 148.0, -0.6)),

        ("rig:RightHandPinky1", "rig:RightHand", (-76.0, 148.0, -1.8)),
        ("rig:RightHandPinky2", "rig:RightHandPinky1", (-78.5, 148.0, -1.8)),
        ("rig:RightHandPinky3", "rig:RightHandPinky2", (-80.6, 148.0, -1.8)),

        # Left Leg Chain & Twist (6 bones)
        ("rig:LeftUpLeg", "rig:Hips", (10.0, 96.0, 0.0)),
        ("rig:LeftUpLegTwist", "rig:LeftUpLeg", (10.0, 74.0, 0.0)),
        ("rig:LeftLeg", "rig:LeftUpLeg", (10.0, 52.0, 0.0)),
        ("rig:LeftFoot", "rig:LeftLeg", (10.0, 10.0, 0.0)),
        ("rig:LeftToeBase", "rig:LeftFoot", (10.0, 3.0, 14.0)),
        ("rig:LeftToe_End", "rig:LeftToeBase", (10.0, 3.0, 22.0)),

        # Right Leg Chain & Twist (6 bones)
        ("rig:RightUpLeg", "rig:Hips", (-10.0, 96.0, 0.0)),
        ("rig:RightUpLegTwist", "rig:RightUpLeg", (-10.0, 74.0, 0.0)),
        ("rig:RightLeg", "rig:RightUpLeg", (-10.0, 52.0, 0.0)),
        ("rig:RightFoot", "rig:RightLeg", (-10.0, 10.0, 0.0)),
        ("rig:RightToeBase", "rig:RightFoot", (-10.0, 3.0, 14.0)),
        ("rig:RightToe_End", "rig:RightToeBase", (-10.0, 3.0, 22.0)),
    ]

    log(f"Defined {len(bone_specs)} canonical bones", log_file)
    assert len(bone_specs) == 61, f"Expected 61 bones, got {len(bone_specs)}"

    for name, parent_name, head in bone_specs:
        b = eb.new(name)
        b.head = head
        b.tail = (head[0], head[1] + 5.0, head[2])
        b.roll = 0.0
        if parent_name:
            b.parent = eb[parent_name]

    bpy.ops.object.mode_set(mode="OBJECT")
    log("Created 61-bone Armature hierarchy with uniform +Y edit bone tail orientation", log_file)

    # 3. Create Articulated Humanoid Surface Mesh (Alpha_Surface)
    def create_body_part(
        shape_type: str,
        loc: tuple[float, float, float],
        scale: tuple[float, float, float],
        rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        if shape_type == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=16, ring_count=12, radius=1.0, location=loc
            )
        elif shape_type == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=16, radius=1.0, depth=1.0, location=loc
            )
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")
        obj = bpy.context.active_object
        obj.scale = scale
        obj.rotation_euler = rot
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        return obj

    parts = []
    # In Blender, default cylinder axis is along Z.
    # To orient along vertical Y-axis (height): rotate 90 deg around X: rot=(math.radians(90.0), 0.0, 0.0)
    # To orient along lateral X-axis (arms): rotate 90 deg around Y: rot=(0.0, math.radians(90.0), 0.0)
    rot_y = (math.radians(90.0), 0.0, 0.0)
    rot_x = (0.0, math.radians(90.0), 0.0)

    # Head & Neck
    parts.append(create_body_part("sphere", (0.0, 168.0, 1.0), (8.0, 10.0, 8.5)))
    parts.append(create_body_part("sphere", (0.0, 161.0, 3.0), (5.5, 6.0, 6.0)))
    parts.append(create_body_part("cylinder", (0.0, 154.0, 0.0), (4.5, 4.5, 12.0), rot=rot_y))

    # Torso (Hips to Shoulders: Y in [94, 144])
    parts.append(create_body_part("sphere", (0.0, 140.0, 0.5), (15.5, 12.0, 10.5)))
    parts.append(create_body_part("cylinder", (0.0, 126.0, 0.0), (12.5, 9.5, 18.0), rot=rot_y))
    parts.append(create_body_part("cylinder", (0.0, 114.0, 0.0), (12.0, 9.0, 18.0), rot=rot_y))
    parts.append(create_body_part("sphere", (0.0, 101.0, 0.0), (14.0, 11.0, 10.0)))
    parts.append(create_body_part("cylinder", (0.0, 94.0, 0.0), (12.5, 9.0, 12.0), rot=rot_y))

    # Arms, Forearms & Articulated 5-Finger Hands (Left + Right)
    for side, sign in [("Left", 1.0), ("Right", -1.0)]:
        # Shoulder / Arm / Elbow / Forearm
        parts.append(create_body_part("sphere", (sign * 18.0, 148.0, 0.0), (6.5, 6.0, 6.0)))
        parts.append(create_body_part("cylinder", (sign * 31.5, 148.0, 0.0), (4.6, 4.6, 27.0), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 45.0, 148.0, 0.0), (4.4, 4.2, 4.2)))
        parts.append(create_body_part("cylinder", (sign * 58.5, 148.0, 0.0), (3.8, 3.6, 27.0), rot=rot_x))
        # Wrist & Contoured Palm Slab
        parts.append(create_body_part("sphere", (sign * 72.0, 148.0, 0.0), (3.2, 2.5, 3.2)))
        parts.append(create_body_part("sphere", (sign * 74.5, 148.0, 0.0), (3.0, 2.0, 3.8)))
        parts.append(create_body_part("sphere", (sign * 76.5, 148.0, 0.0), (2.0, 1.8, 4.0)))

        # 5 Articulated Digits per hand:
        # Thumb: carpometacarpal, proximal phalanx, distal phalanx (pointing forward-laterally)
        parts.append(create_body_part("cylinder", (sign * 74.5, 147.2, 2.2), (1.0, 1.0, 2.5)))
        parts.append(create_body_part("sphere", (sign * 75.8, 147.0, 3.0), (1.0, 1.0, 1.0)))
        parts.append(create_body_part("cylinder", (sign * 77.2, 146.8, 3.8), (0.9, 0.9, 2.4)))
        parts.append(create_body_part("sphere", (sign * 78.5, 146.6, 4.5), (0.85, 0.85, 0.85)))
        parts.append(create_body_part("sphere", (sign * 79.5, 146.4, 4.8), (1.0, 0.48, 0.48)))

        # Index: MCP knuckle, proximal phalanx, PIP joint, intermediate phalanx, DIP joint, tip
        parts.append(create_body_part("sphere", (sign * 76.8, 148.0, 1.8), (0.70, 0.50, 0.45)))
        parts.append(create_body_part("cylinder", (sign * 78.2, 148.0, 1.8), (0.45, 0.45, 2.4), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 79.5, 148.0, 1.8), (0.65, 0.45, 0.45)))
        parts.append(create_body_part("cylinder", (sign * 80.8, 148.0, 1.8), (0.42, 0.42, 2.2), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 82.0, 148.0, 1.8), (0.60, 0.42, 0.42)))
        parts.append(create_body_part("sphere", (sign * 83.2, 148.0, 1.8), (1.2, 0.45, 0.45)))

        # Middle: MCP knuckle, proximal phalanx, PIP joint, intermediate phalanx, DIP joint, tip
        parts.append(create_body_part("sphere", (sign * 77.0, 148.0, 0.6), (0.70, 0.52, 0.45)))
        parts.append(create_body_part("cylinder", (sign * 78.5, 148.0, 0.6), (0.45, 0.45, 2.6), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 80.0, 148.0, 0.6), (0.70, 0.46, 0.45)))
        parts.append(create_body_part("cylinder", (sign * 81.4, 148.0, 0.6), (0.44, 0.44, 2.4), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 82.8, 148.0, 0.6), (0.65, 0.44, 0.44)))
        parts.append(create_body_part("sphere", (sign * 84.0, 148.0, 0.6), (1.4, 0.48, 0.48)))

        # Ring: MCP knuckle, proximal phalanx, PIP joint, intermediate phalanx, DIP joint, tip
        parts.append(create_body_part("sphere", (sign * 76.6, 148.0, -0.6), (0.70, 0.48, 0.44)))
        parts.append(create_body_part("cylinder", (sign * 78.0, 148.0, -0.6), (0.44, 0.44, 2.4), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 79.4, 148.0, -0.6), (0.65, 0.44, 0.44)))
        parts.append(create_body_part("cylinder", (sign * 80.6, 148.0, -0.6), (0.42, 0.42, 2.2), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 81.8, 148.0, -0.6), (0.60, 0.42, 0.42)))
        parts.append(create_body_part("sphere", (sign * 82.9, 148.0, -0.6), (1.2, 0.44, 0.44)))

        # Pinky: MCP knuckle, proximal phalanx, PIP joint, intermediate phalanx, DIP joint, tip
        parts.append(create_body_part("sphere", (sign * 76.0, 148.0, -1.8), (0.65, 0.45, 0.40)))
        parts.append(create_body_part("cylinder", (sign * 77.2, 148.0, -1.8), (0.40, 0.40, 2.2), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 78.5, 148.0, -1.8), (0.60, 0.40, 0.40)))
        parts.append(create_body_part("cylinder", (sign * 79.6, 148.0, -1.8), (0.38, 0.38, 2.0), rot=rot_x))
        parts.append(create_body_part("sphere", (sign * 80.6, 148.0, -1.8), (0.55, 0.38, 0.38)))
        parts.append(create_body_part("sphere", (sign * 81.6, 148.0, -1.8), (1.0, 0.40, 0.40)))

    # Legs & Feet (Left + Right)
    for side, sign in [("Left", 1.0), ("Right", -1.0)]:
        parts.append(create_body_part("sphere", (sign * 10.0, 96.0, 0.0), (7.5, 7.5, 7.0)))
        parts.append(create_body_part("cylinder", (sign * 10.0, 74.0, 0.0), (6.5, 6.5, 46.0), rot=rot_y))
        parts.append(create_body_part("sphere", (sign * 10.0, 52.0, 0.5), (5.5, 5.8, 5.5)))
        parts.append(create_body_part("cylinder", (sign * 10.0, 31.0, -0.5), (5.0, 5.2, 44.0), rot=rot_y))
        parts.append(create_body_part("sphere", (sign * 10.0, 10.0, 0.0), (4.2, 4.2, 4.2)))
        parts.append(create_body_part("sphere", (sign * 10.0, 5.0, -3.5), (4.0, 4.0, 5.0)))
        parts.append(create_body_part("cylinder", (sign * 10.0, 4.5, 5.0), (4.0, 3.5, 14.0)))
        parts.append(create_body_part("sphere", (sign * 10.0, 3.8, 14.0), (4.5, 3.2, 4.2)))
        parts.append(create_body_part("cylinder", (sign * 10.0, 3.2, 18.0), (4.2, 2.8, 8.0)))
        parts.append(create_body_part("sphere", (sign * 10.0, 3.0, 22.0), (3.8, 2.5, 3.5)))

    # Join body parts into single object
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    surface_obj = bpy.context.active_object
    surface_obj.name = "Alpha_Surface"
    surface_obj.data.name = "Alpha_Surface"

    # Remesh at 0.90 cm voxel size to preserve clean finger separation without web-bridging
    surface_obj.data.remesh_voxel_size = 0.90
    bpy.ops.object.voxel_remesh()

    mod_smooth = surface_obj.modifiers.new(name="Smooth", type="SMOOTH")
    mod_smooth.factor = 0.4
    mod_smooth.iterations = 2
    bpy.ops.object.modifier_apply(modifier="Smooth")
    bpy.ops.object.shade_smooth()

    bpy.context.view_layer.objects.active = surface_obj
    bpy.ops.object.select_all(action="DESELECT")
    surface_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    log(
        f"Generated continuous Alpha_Surface mesh: {len(surface_obj.data.vertices)} vertices, "
        f"{len(surface_obj.data.polygons)} polygons",
        log_file,
    )

    # 4. Construct Articulated Accent Rings (Alpha_Joints)
    joints_parts = []
    rot_ring_y = (math.radians(90.0), 0.0, 0.0)
    rot_ring_x = (0.0, math.radians(90.0), 0.0)
    rings_spec: list[tuple[tuple[float, float, float], float, float, tuple[float, float, float], str]] = [
        ((0.0, 154.0, 0.0), 4.7, 0.5, rot_ring_y, "rig:Neck"),
        ((0.0, 114.0, 0.0), 10.8, 0.7, rot_ring_y, "rig:Spine"),
        ((16.0, 148.0, 0.0), 5.6, 0.5, rot_ring_x, "rig:LeftShoulder"),
        ((-16.0, 148.0, 0.0), 5.6, 0.5, rot_ring_x, "rig:RightShoulder"),
        ((44.0, 148.0, 0.0), 4.2, 0.5, rot_ring_x, "rig:LeftArm"),
        ((-44.0, 148.0, 0.0), 4.2, 0.5, rot_ring_x, "rig:RightArm"),
        ((71.0, 148.0, 0.0), 3.0, 0.4, rot_ring_x, "rig:LeftForeArm"),
        ((-71.0, 148.0, 0.0), 3.0, 0.4, rot_ring_x, "rig:RightForeArm"),
        ((10.0, 52.0, 0.0), 5.3, 0.5, rot_ring_y, "rig:LeftUpLeg"),
        ((-10.0, 52.0, 0.0), 5.3, 0.5, rot_ring_y, "rig:RightUpLeg"),
        ((10.0, 12.0, 0.0), 3.7, 0.4, rot_ring_y, "rig:LeftLeg"),
        ((-10.0, 12.0, 0.0), 3.7, 0.4, rot_ring_y, "rig:RightLeg"),
    ]

    for loc, major_r, minor_r, rot, bname in rings_spec:
        bpy.ops.mesh.primitive_torus_add(
            major_radius=major_r,
            minor_radius=minor_r,
            major_segments=20,
            minor_segments=8,
            location=loc,
        )
        ring_obj = bpy.context.active_object
        ring_obj.rotation_euler = rot
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # Assign 100% rigid weight to host articulation bone before joining
        vg = ring_obj.vertex_groups.new(name=bname)
        vg.add(list(range(len(ring_obj.data.vertices))), 1.0, "REPLACE")
        joints_parts.append(ring_obj)

    bpy.ops.object.select_all(action="DESELECT")
    for jp in joints_parts:
        jp.select_set(True)
    bpy.context.view_layer.objects.active = joints_parts[0]
    bpy.ops.object.join()
    joints_obj = bpy.context.active_object
    joints_obj.name = "Alpha_Joints"
    joints_obj.data.name = "Alpha_Joints"
    bpy.ops.object.shade_smooth()

    bpy.context.view_layer.objects.active = joints_obj
    bpy.ops.object.select_all(action="DESELECT")
    joints_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    log(
        f"Generated articulated Alpha_Joints mesh: {len(joints_obj.data.vertices)} vertices, "
        f"{len(joints_obj.data.polygons)} polygons across 12 accent rings",
        log_file,
    )

    # 5. UV Unwrapping
    for obj in [surface_obj, joints_obj]:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
    log("Performed conformal UV smart project unwrapping on both meshes", log_file)

    # 6. Studio Gradient Textures & PBR Materials
    w_tex, h_tex = 512, 512
    # Slate-to-pearl studio gradient for Alpha_Surface
    img_surf = bpy.data.images.new("Articulated_Surface_Tex", width=w_tex, height=h_tex, alpha=False)
    pix_surf = np.zeros((h_tex, w_tex, 4), dtype=np.float32)
    for y_idx in range(h_tex):
        v_coord = y_idx / (h_tex - 1)
        for x_idx in range(w_tex):
            pix_surf[y_idx, x_idx] = [
                0.70 + 0.18 * v_coord,
                0.72 + 0.18 * v_coord,
                0.78 + 0.16 * v_coord,
                1.0,
            ]
    img_surf.pixels.foreach_set(pix_surf.flatten())
    img_surf.pack()

    mat_surf = bpy.data.materials.new(name="Alpha_Surface_Mat")
    mat_surf.use_nodes = True
    bsdf_surf = mat_surf.node_tree.nodes.get("Principled BSDF")
    if bsdf_surf:
        if "Metallic" in bsdf_surf.inputs:
            bsdf_surf.inputs["Metallic"].default_value = 0.05
        if "Roughness" in bsdf_surf.inputs:
            bsdf_surf.inputs["Roughness"].default_value = 0.35
    tex_surf = mat_surf.node_tree.nodes.new(type="ShaderNodeTexImage")
    tex_surf.image = img_surf
    mat_surf.node_tree.links.new(tex_surf.outputs["Color"], bsdf_surf.inputs["Base Color"])
    surface_obj.data.materials.append(mat_surf)

    # Dark titanium studio gradient for Alpha_Joints
    img_joints = bpy.data.images.new("Articulated_Joints_Tex", width=w_tex, height=h_tex, alpha=False)
    pix_joints = np.zeros((h_tex, w_tex, 4), dtype=np.float32)
    for y_idx in range(h_tex):
        v_coord = y_idx / (h_tex - 1)
        for x_idx in range(w_tex):
            pix_joints[y_idx, x_idx] = [
                0.35 + 0.08 * v_coord,
                0.37 + 0.08 * v_coord,
                0.40 + 0.08 * v_coord,
                1.0,
            ]
    img_joints.pixels.foreach_set(pix_joints.flatten())
    img_joints.pack()

    mat_joints = bpy.data.materials.new(name="Alpha_Joints_Mat")
    mat_joints.use_nodes = True
    bsdf_joints = mat_joints.node_tree.nodes.get("Principled BSDF")
    if bsdf_joints:
        if "Metallic" in bsdf_joints.inputs:
            bsdf_joints.inputs["Metallic"].default_value = 0.60
        if "Roughness" in bsdf_joints.inputs:
            bsdf_joints.inputs["Roughness"].default_value = 0.30
    tex_joints = mat_joints.node_tree.nodes.new(type="ShaderNodeTexImage")
    tex_joints.image = img_joints
    mat_joints.node_tree.links.new(tex_joints.outputs["Color"], bsdf_joints.inputs["Base Color"])
    joints_obj.data.materials.append(mat_joints)
    log("Embedded studio gradient texture maps and configured PBR materials", log_file)

    # 7. Compute Analytical Skin Weights for all 61 bones on Alpha_Surface
    all_bone_names = [b[0] for b in bone_specs]
    n_surf_verts = len(surface_obj.data.vertices)
    surf_coords = np.empty((n_surf_verts, 3), dtype=np.float64)
    surface_obj.data.vertices.foreach_get("co", surf_coords.reshape(-1))

    weights: dict[str, np.ndarray] = {b: np.zeros(n_surf_verts, dtype=np.float64) for b in all_bone_names}

    for i in range(n_surf_verts):
        xi, yi, zi = surf_coords[i]
        abs_x = abs(xi)
        side = "Left" if xi >= 0 else "Right"

        # 1. Lower Body: Pelvis, Thighs, Legs, Feet (yi < 92.0 cm)
        if yi < 92.0:
            if yi <= 74.0:
                if yi > 58.0:
                    # Upper leg vs Upper leg twist (femoral shaft twist sharing)
                    t_tw = float(smoothstep(58.0, 85.0, yi))
                    weights[f"rig:{side}UpLeg"][i] = 1.0 - 0.5 * t_tw
                    weights[f"rig:{side}UpLegTwist"][i] = 0.5 * t_tw
                elif yi >= 46.0:
                    t = float(smoothstep(46.0, 58.0, yi))
                    weights[f"rig:{side}UpLeg"][i] = t
                    weights[f"rig:{side}Leg"][i] = 1.0 - t
                elif yi > 16.0:
                    weights[f"rig:{side}Leg"][i] = 1.0
                elif yi >= 8.0:
                    t = float(smoothstep(8.0, 16.0, yi))
                    weights[f"rig:{side}Leg"][i] = t
                    weights[f"rig:{side}Foot"][i] = 1.0 - t
                else:
                    if zi <= 10.0:
                        weights[f"rig:{side}Foot"][i] = 1.0
                    elif zi <= 16.0:
                        t = float(smoothstep(10.0, 16.0, zi))
                        weights[f"rig:{side}ToeBase"][i] = t
                        weights[f"rig:{side}Foot"][i] = 1.0 - t
                    else:
                        weights[f"rig:{side}ToeBase"][i] = 1.0
                continue

            # Hip / Pelvis crease (74.0 <= yi < 92.0 cm)
            t_blend = float(smoothstep(74.0, 88.0, yi))
            sx = float(smoothstep(2.2, 3.2, abs_x))
            w_leg = sx * (1.0 - t_blend)
            weights[f"rig:{side}UpLeg"][i] = w_leg
            weights["rig:Hips"][i] = 1.0 - w_leg
            continue

        # 2. Head & Neck column (yi >= 144.0 cm and abs_x <= 6.5 cm)
        if yi >= 144.0 and abs_x <= 6.5:
            if yi >= 158.0:
                t = float(smoothstep(158.0, 164.0, yi))
                weights["rig:Head"][i] = t
                weights["rig:Neck"][i] = 1.0 - t
            elif yi >= 152.0:
                weights["rig:Neck"][i] = 1.0
            else:
                sx = float(smoothstep(6.5, 3.5, abs_x))
                sy = float(smoothstep(144.0, 150.0, yi))
                m_neck = sx * sy
                weights["rig:Neck"][i] = m_neck
                weights["rig:Spine2"][i] = 1.0 - m_neck
            continue

        if yi >= 158.0:
            if yi >= 164.0:
                weights["rig:Head"][i] = 1.0
            else:
                t = float(smoothstep(158.0, 164.0, yi))
                weights["rig:Head"][i] = t
                weights["rig:Neck"][i] = 1.0 - t
            continue

        # 3. Arms, Forearms, Hands & Articulated 5 Fingers (abs_x >= 8.0 cm and yi >= 126.0 cm)
        if abs_x >= 8.0 and yi >= 126.0:
            sx = float(smoothstep(8.0, 13.5, abs_x))
            sy = float(smoothstep(126.0, 138.0, yi))
            m_arm = sx * sy

            if abs_x >= 17.0:
                # Arm / Twists / Forearm / Hand / Fingers
                if abs_x >= 76.0:
                    # Individual 5-finger articulated digits
                    # Thumb partition: front-facing (+Z) angled downward
                    if zi >= 2.0 and abs_x <= 80.5:
                        if abs_x < 76.5:
                            t = float(smoothstep(75.5, 76.5, abs_x))
                            weights[f"rig:{side}HandThumb1"][i] = t
                            weights[f"rig:{side}Hand"][i] = 1.0 - t
                        elif abs_x < 78.0:
                            t = float(smoothstep(76.5, 78.0, abs_x))
                            weights[f"rig:{side}HandThumb2"][i] = t
                            weights[f"rig:{side}HandThumb1"][i] = 1.0 - t
                        else:
                            t = float(smoothstep(78.0, 79.5, abs_x))
                            weights[f"rig:{side}HandThumb3"][i] = t
                            weights[f"rig:{side}HandThumb2"][i] = 1.0 - t
                    else:
                        # 4 Parallel fingers partitioned along anterior-posterior Z axis
                        if zi >= 1.2:
                            f_name = "Index"
                        elif zi >= 0.0:
                            f_name = "Middle"
                        elif zi >= -1.2:
                            f_name = "Ring"
                        else:
                            f_name = "Pinky"

                        if abs_x < 77.5:
                            t = float(smoothstep(76.0, 77.5, abs_x))
                            weights[f"rig:{side}Hand{f_name}1"][i] = t
                            weights[f"rig:{side}Hand"][i] = 1.0 - t
                        elif abs_x < 80.0:
                            t = float(smoothstep(77.5, 80.0, abs_x))
                            weights[f"rig:{side}Hand{f_name}2"][i] = t
                            weights[f"rig:{side}Hand{f_name}1"][i] = 1.0 - t
                        elif abs_x < 82.0:
                            t = float(smoothstep(80.0, 82.0, abs_x))
                            weights[f"rig:{side}Hand{f_name}3"][i] = t
                            weights[f"rig:{side}Hand{f_name}2"][i] = 1.0 - t
                        else:
                            weights[f"rig:{side}Hand{f_name}3"][i] = 1.0

                elif abs_x >= 71.5:
                    # Hand palm
                    t = float(smoothstep(68.0, 72.0, abs_x))
                    weights[f"rig:{side}Hand"][i] = t
                    weights[f"rig:{side}ForeArm"][i] = 1.0 - t
                elif abs_x >= 48.0:
                    # Forearm vs Forearm Twist (radius/ulna pronation/supination sharing)
                    t_tw = float(smoothstep(51.0, 65.0, abs_x))
                    weights[f"rig:{side}ForeArm"][i] = 1.0 - 0.5 * t_tw
                    weights[f"rig:{side}ForeArmTwist"][i] = 0.5 * t_tw
                elif abs_x >= 41.0:
                    t = float(smoothstep(41.0, 48.0, abs_x))
                    weights[f"rig:{side}ForeArm"][i] = t
                    weights[f"rig:{side}Arm"][i] = 1.0 - t
                elif abs_x >= 28.0:
                    # Upper arm vs Upper arm Twist (humeral twist sharing)
                    t_tw = float(smoothstep(24.0, 38.0, abs_x))
                    weights[f"rig:{side}Arm"][i] = 1.0 - 0.5 * t_tw
                    weights[f"rig:{side}ArmTwist"][i] = 0.5 * t_tw
                else:
                    t = float(smoothstep(17.0, 28.0, abs_x))
                    weights[f"rig:{side}Arm"][i] = t
                    weights[f"rig:{side}Shoulder"][i] = 1.0 - t
                continue

            weights[f"rig:{side}Shoulder"][i] = m_arm
            weights["rig:Spine2"][i] = 1.0 - m_arm
            continue

        # 4. Central Torso / Spine Column (92.0 <= yi < 144.0 cm)
        if yi >= 124.0:
            weights["rig:Spine2"][i] = 1.0
        elif yi >= 118.0:
            t = float(smoothstep(118.0, 124.0, yi))
            weights["rig:Spine2"][i] = t
            weights["rig:Spine1"][i] = 1.0 - t
        elif yi >= 112.0:
            weights["rig:Spine1"][i] = 1.0
        elif yi >= 106.0:
            t = float(smoothstep(106.0, 112.0, yi))
            weights["rig:Spine1"][i] = t
            weights["rig:Spine"][i] = 1.0 - t
        elif yi >= 100.0:
            weights["rig:Spine"][i] = 1.0
        elif yi >= 95.0:
            t = float(smoothstep(95.0, 100.0, yi))
            weights["rig:Spine"][i] = t
            weights["rig:Hips"][i] = 1.0 - t
        else:
            weights["rig:Hips"][i] = 1.0

    # Normalise weights to strictly enforce partition of unity (sum == 1.0)
    weight_matrix = np.column_stack([weights[b] for b in all_bone_names])
    row_sums = weight_matrix.sum(axis=1)
    zero_rows = np.where(row_sums < 1e-6)[0]
    if len(zero_rows) > 0:
        log(f"Warning: {len(zero_rows)} vertices with zero weight sum; assigning to Hips", log_file)
        for idx in zero_rows:
            weights["rig:Hips"][idx] = 1.0
        weight_matrix = np.column_stack([weights[b] for b in all_bone_names])
        row_sums = weight_matrix.sum(axis=1)

    weight_matrix /= row_sums[:, None]
    max_dev = float(np.max(np.abs(weight_matrix.sum(axis=1) - 1.0)))
    log(f"Alpha_Surface partition of unity max deviation: {max_dev:.3e}", log_file)

    for b_idx, bname in enumerate(all_bone_names):
        weights[bname] = weight_matrix[:, b_idx]

    # Create vertex groups for all 61 bones on both objects
    for bname in all_bone_names:
        if bname not in surface_obj.vertex_groups:
            surface_obj.vertex_groups.new(name=bname)
        if bname not in joints_obj.vertex_groups:
            joints_obj.vertex_groups.new(name=bname)

    # Assign non-zero weights on Alpha_Surface
    for bname, w_arr in weights.items():
        nonzero_idx = np.where(w_arr > 1e-4)[0]
        if len(nonzero_idx) == 0:
            continue
        vg = surface_obj.vertex_groups[bname]
        for idx in nonzero_idx:
            vg.add([int(idx)], float(w_arr[idx]), "REPLACE")

    log("Assigned skin weights for all 61 bones across Alpha_Surface and Alpha_Joints", log_file)

    # 8. Parent meshes to Armature and bind Armature modifiers
    for obj in [surface_obj, joints_obj]:
        obj.parent = arm_obj
        mod = obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm_obj
    log("Parented meshes to Armature and bound Armature skinning modifiers", log_file)

    # 9. Export scene to binary FBX (Kaydara FBX Binary) with embedded textures
    bpy.ops.export_scene.fbx(
        filepath=str(output_fbx),
        check_existing=False,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Z",
        axis_up="Y",
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="NULL",
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
    )

    file_size = output_fbx.stat().st_size
    log(f"Exported FBX binary successfully: {output_fbx} ({file_size} bytes)", log_file)
    log("STATUS: SUCCESS", log_file)
    return True


def main() -> int:
    """CLI entry point for generator execution within Blender Python."""
    default_output = (
        Path(__file__).resolve().parent.parent / "assets" / "rigs" / "humanoid_articulated.fbx"
    )
    default_log = default_output.with_suffix(".log")

    parser = argparse.ArgumentParser(
        description="Generate CC0 61-Bone Articulated Humanoid Character Rig FBX"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help="Destination path for output humanoid_articulated.fbx",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=str(default_log),
        help="Destination path for execution log file",
    )

    # Support arguments passed after '--' in Blender CLI or direct flags
    if "--" in sys.argv:
        cli_args = sys.argv[sys.argv.index("--") + 1 :]
        parsed = parser.parse_args(cli_args)
    else:
        explicit_args: list[str] = []
        for i, arg in enumerate(sys.argv):
            if arg in ("--output", "--log") and i + 1 < len(sys.argv):
                explicit_args.extend([arg, sys.argv[i + 1]])
        parsed = parser.parse_args(explicit_args)

    out_path = Path(parsed.output).resolve()
    log_path = Path(parsed.log).resolve() if parsed.log else out_path.with_suffix(".log")

    try:
        success = build_articulated_humanoid_fbx(out_path, log_path)
        return 0 if success else 1
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL ERROR: {exc}\n{traceback.format_exc()}", log_path)
        log("STATUS: FAILED", log_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())

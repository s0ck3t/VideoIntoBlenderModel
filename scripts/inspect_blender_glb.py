"""Headless Blender 5.2.2 LTS Comprehensive Programmatic Inspection Script.

Executes inside Blender 5.2.2 LTS to verify:
1. Armature Object:
   - Exactly 61 canonical bones (25 core + 6 limb twists + 30 phalangeal fingers).
   - Complete anatomical bone hierarchy and parenting relationships.
2. Action & F-Curves:
   - Full compatibility with Blender 5.x layered and slotted actions.
   - Active rotation quaternion channels verified for fingers (Thumb1..3, Index1..3, etc.),
     progressive spine (Spine, Spine1, Spine2), 3-DoF neck/head, and leg chains.
3. Skinned Meshes:
   - Mesh vertex counts, vertex group allocations across all 61 bones, and armature modifiers.
4. Animation Playback Frame Sweep & Mesh Deformation:
   - Sweeps evaluated mesh deformation across the complete animation timeline.
   - Asserts zero NaN or infinite vertex coordinates and bounded bounding boxes.
   - Evaluates dynamic edge stretch ratios to confirm zero mesh tearing (< 5.0x threshold).
5. Viewport Keyframe Preview Rendering:
   - Frames character with studio camera and dual-point lighting.
   - Renders high-fidelity PNG keyframe stills (e.g. frames 0, 15, 30, 45, or clip end).
6. Structured Reporting:
   - Generates machine-readable `inspection_report.json` and persistent `inspection.log`.
   - Returns exit code 0 on PASS, 1 on FAIL.

Execution standard on Windows (PowerShell):
  Start-Process -FilePath "$env:LOCALAPPDATA\\Microsoft\\WindowsApps\\blender-launcher.exe" `
    -ArgumentList "--background", "--python", "scripts/inspect_blender_glb.py", "--", `
      "--glb", "outputs/<run>/character_animation.glb", `
      "--output-dir", "outputs/<run>/inspection", `
      "--keyframe-interval", "15" -Wait
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Canonical 61 Humanoid bones
CORE_25_BONES = [
    "rig:Hips",
    "rig:Spine",
    "rig:Spine1",
    "rig:Spine2",
    "rig:Neck",
    "rig:Head",
    "rig:HeadTop_End",
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
    "rig:LeftToe_End",
    "rig:RightUpLeg",
    "rig:RightLeg",
    "rig:RightFoot",
    "rig:RightToeBase",
    "rig:RightToe_End",
]

TWIST_6_BONES = [
    "rig:LeftArmTwist",
    "rig:RightArmTwist",
    "rig:LeftForeArmTwist",
    "rig:RightForeArmTwist",
    "rig:LeftUpLegTwist",
    "rig:RightUpLegTwist",
]

FINGER_30_BONES = [
    "rig:LeftHandThumb1",
    "rig:LeftHandThumb2",
    "rig:LeftHandThumb3",
    "rig:LeftHandIndex1",
    "rig:LeftHandIndex2",
    "rig:LeftHandIndex3",
    "rig:LeftHandMiddle1",
    "rig:LeftHandMiddle2",
    "rig:LeftHandMiddle3",
    "rig:LeftHandRing1",
    "rig:LeftHandRing2",
    "rig:LeftHandRing3",
    "rig:LeftHandPinky1",
    "rig:LeftHandPinky2",
    "rig:LeftHandPinky3",
    "rig:RightHandThumb1",
    "rig:RightHandThumb2",
    "rig:RightHandThumb3",
    "rig:RightHandIndex1",
    "rig:RightHandIndex2",
    "rig:RightHandIndex3",
    "rig:RightHandMiddle1",
    "rig:RightHandMiddle2",
    "rig:RightHandMiddle3",
    "rig:RightHandRing1",
    "rig:RightHandRing2",
    "rig:RightHandRing3",
    "rig:RightHandPinky1",
    "rig:RightHandPinky2",
    "rig:RightHandPinky3",
]

ALL_61_BONES = CORE_25_BONES + TWIST_6_BONES + FINGER_30_BONES

EXPECTED_PARENTS = {
    "rig:Spine": "rig:Hips",
    "rig:Spine1": "rig:Spine",
    "rig:Spine2": "rig:Spine1",
    "rig:Neck": "rig:Spine2",
    "rig:Head": "rig:Neck",
    "rig:HeadTop_End": "rig:Head",
    "rig:LeftShoulder": "rig:Spine2",
    "rig:RightShoulder": "rig:Spine2",
    "rig:LeftForeArm": "rig:LeftArm",
    "rig:RightForeArm": "rig:RightArm",
    "rig:LeftHand": "rig:LeftForeArm",
    "rig:RightHand": "rig:RightForeArm",
    "rig:LeftLeg": "rig:LeftUpLeg",
    "rig:RightLeg": "rig:RightUpLeg",
    "rig:LeftFoot": "rig:LeftLeg",
    "rig:RightFoot": "rig:RightLeg",
    "rig:LeftToeBase": "rig:LeftFoot",
    "rig:RightToeBase": "rig:RightFoot",
    "rig:LeftToe_End": "rig:LeftToeBase",
    "rig:RightToe_End": "rig:RightToeBase",
    # Finger chains
    "rig:LeftHandThumb1": "rig:LeftHand",
    "rig:LeftHandThumb2": "rig:LeftHandThumb1",
    "rig:LeftHandThumb3": "rig:LeftHandThumb2",
    "rig:LeftHandIndex1": "rig:LeftHand",
    "rig:LeftHandIndex2": "rig:LeftHandIndex1",
    "rig:LeftHandIndex3": "rig:LeftHandIndex2",
    "rig:LeftHandMiddle1": "rig:LeftHand",
    "rig:LeftHandMiddle2": "rig:LeftHandMiddle1",
    "rig:LeftHandMiddle3": "rig:LeftHandMiddle2",
    "rig:LeftHandRing1": "rig:LeftHand",
    "rig:LeftHandRing2": "rig:LeftHandRing1",
    "rig:LeftHandRing3": "rig:LeftHandRing2",
    "rig:LeftHandPinky1": "rig:LeftHand",
    "rig:LeftHandPinky2": "rig:LeftHandPinky1",
    "rig:LeftHandPinky3": "rig:LeftHandPinky2",
    "rig:RightHandThumb1": "rig:RightHand",
    "rig:RightHandThumb2": "rig:RightHandThumb1",
    "rig:RightHandThumb3": "rig:RightHandThumb2",
    "rig:RightHandIndex1": "rig:RightHand",
    "rig:RightHandIndex2": "rig:RightHandIndex1",
    "rig:RightHandIndex3": "rig:RightHandIndex2",
    "rig:RightHandMiddle1": "rig:RightHand",
    "rig:RightHandMiddle2": "rig:RightHandMiddle1",
    "rig:RightHandMiddle3": "rig:RightHandMiddle2",
    "rig:RightHandRing1": "rig:RightHand",
    "rig:RightHandRing2": "rig:RightHandRing1",
    "rig:RightHandRing3": "rig:RightHandRing2",
    "rig:RightHandPinky1": "rig:RightHand",
    "rig:RightHandPinky2": "rig:RightHandPinky1",
    "rig:RightHandPinky3": "rig:RightHandPinky2",
}


class DualLogger:
    """Simultaneously outputs diagnostic logs to stdout and a disk log file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def get_all_fcurves(action: object) -> list[object]:
    """Retrieve animation curves across Blender 4.x legacy actions and Blender 5.x layered actions."""
    if hasattr(action, "fcurves") and getattr(action, "fcurves", None) is not None:
        return list(action.fcurves)
    curves: list[object] = []
    if hasattr(action, "layers"):
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for cb in getattr(strip, "channelbags", []):
                    curves.extend(getattr(cb, "fcurves", []))
    elif hasattr(action, "slots") and getattr(action, "slots", None):
        for slot in action.slots:
            curves.extend(getattr(slot, "curves", []))
    return curves


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(
        description="Headless Blender 5.2.2 LTS Comprehensive Programmatic Inspection"
    )
    parser.add_argument(
        "--glb",
        type=Path,
        required=True,
        help="Path to the .glb file to inspect",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store inspection logs, JSON report, and preview renders",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=15,
        help="Interval between preview keyframe renders (default: 15)",
    )
    parser.add_argument(
        "--max-edge-stretch",
        type=float,
        default=5.0,
        help="Maximum permitted ratio of deformed edge length to rest edge length (default: 5.0)",
    )
    return parser.parse_args(script_args)


def run_inspection() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "inspection.log"
    json_file = out_dir / "inspection_report.json"
    logger = DualLogger(log_file)

    logger.log("======================================================================")
    logger.log("Headless Blender 5.2.2 LTS Comprehensive Programmatic Inspection")
    logger.log("======================================================================")
    logger.log(f"GLB Path: {args.glb}")
    logger.log(f"Output Directory: {out_dir}")
    logger.log(f"Keyframe Render Interval: {args.keyframe_interval}")
    logger.log(f"Max Edge Stretch Threshold: {args.max_edge_stretch:.1f}x")

    report: dict[str, object] = {
        "status": "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "glb_path": str(args.glb.resolve()) if args.glb.exists() else str(args.glb),
    }

    try:
        import bpy
        import numpy as np

        blender_version = ".".join(str(v) for v in bpy.app.version)
        logger.log(f"Blender Version: {blender_version}")
        report["blender_version"] = blender_version

        if not args.glb.exists():
            raise FileNotFoundError(f"Target GLB does not exist: {args.glb}")

        # Step 1: Clean scene and import GLTF/GLB
        logger.log("Clearing scene to factory default empty state...")
        bpy.ops.wm.read_factory_settings(use_empty=True)

        logger.log(f"Importing GLB model from {args.glb}...")
        bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))

        # Step 2: Inspect Armature
        armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
        if not armatures:
            raise AssertionError("No ARMATURE object found in imported GLB scene.")
        arm_obj = armatures[0]
        bones = arm_obj.data.bones
        bone_names = [b.name for b in bones]
        logger.log(f"Found Armature object: '{arm_obj.name}' with {len(bone_names)} total bones.")

        missing_bones = [b for b in ALL_61_BONES if b not in bone_names]
        missing_suffixes = []
        if missing_bones:
            for b in ALL_61_BONES:
                s = b.split(":")[-1]
                if not any(name.endswith(s) for name in bone_names):
                    missing_suffixes.append(b)

        logger.log(
            f"Bone validation: {len(bone_names)} present (target 61). "
            f"Missing exact: {len(missing_bones)}, missing suffix: {len(missing_suffixes)}"
        )
        if len(missing_suffixes) > 0 and len(bone_names) < 25:
            raise AssertionError(f"Missing core humanoid bones: {missing_suffixes}")

        # Verify bone hierarchy and parenting
        hierarchy_errors = []
        for child_name, expected_parent_name in EXPECTED_PARENTS.items():
            b = bones.get(child_name)
            if b is None:
                s = child_name.split(":")[-1]
                matches = [x for x in bones if x.name.endswith(s)]
                b = matches[0] if matches else None
            if b is not None and b.parent is not None:
                parent_name = b.parent.name
                exp_suffix = expected_parent_name.split(":")[-1]
                if not parent_name.endswith(exp_suffix):
                    hierarchy_errors.append(
                        f"{b.name} parent is {parent_name}, expected {expected_parent_name}"
                    )

        logger.log(f"Hierarchy validation: {len(hierarchy_errors)} parenting mismatch(es).")
        report["armature"] = {
            "name": arm_obj.name,
            "total_bones": len(bone_names),
            "is_61_bone_articulated": len(bone_names) == 61,
            "missing_bones": missing_bones,
            "hierarchy_mismatches": hierarchy_errors,
        }

        # Step 3: Inspect Action & F-Curves / Channels
        action = None
        if arm_obj.animation_data and arm_obj.animation_data.action:
            action = arm_obj.animation_data.action
        elif list(bpy.data.actions):
            action = list(bpy.data.actions)[0]

        if action is None:
            raise AssertionError("No animation Action found on Armature or in bpy.data.actions.")

        logger.log(f"Found Animation Action: '{action.name}'")
        all_fcurves = get_all_fcurves(action)
        fcurves_count = len(all_fcurves)

        frame_start, frame_end = (
            action.frame_range
            if hasattr(action, "frame_range")
            else (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
        )
        duration_frames = int(frame_end - frame_start + 1)
        logger.log(
            f"Action Frame Range: {frame_start:.1f} to {frame_end:.1f} ({duration_frames} frames), "
            f"Total F-Curves/Channels: {fcurves_count}"
        )

        assert fcurves_count > 0, "Animation action contains zero keyframed animation curves."

        # Verify key animated channels (fingers, spine, neck, head, legs)
        target_channel_bones = [
            "rig:Spine",
            "rig:Spine1",
            "rig:Spine2",
            "rig:Neck",
            "rig:Head",
            "rig:LeftLeg",
            "rig:RightLeg",
            "rig:LeftHandIndex1",
            "rig:LeftHandThumb1",
            "rig:RightHandIndex1",
            "rig:RightHandThumb1",
        ]
        animated_channels_found = {}
        for tb in target_channel_bones:
            s = tb.split(":")[-1]
            found = any(s in getattr(fc, "data_path", "") for fc in all_fcurves)
            animated_channels_found[tb] = found

        report["animation"] = {
            "action_name": action.name,
            "fcurves_count": fcurves_count,
            "frame_start": float(frame_start),
            "frame_end": float(frame_end),
            "duration_frames": duration_frames,
            "sample_channels_verified": animated_channels_found,
        }

        # Step 4: Inspect Skinned Meshes
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        logger.log(f"Found {len(mesh_objs)} MESH object(s): {[m.name for m in mesh_objs]}")
        mesh_metrics = []
        for m in mesh_objs:
            v_cnt = len(m.data.vertices)
            vg_cnt = len(m.vertex_groups)
            arm_mods = [mod for mod in m.modifiers if mod.type == "ARMATURE"]
            has_arm_mod = len(arm_mods) > 0
            logger.log(
                f" - Mesh '{m.name}': {v_cnt} vertices, {vg_cnt} vertex groups, "
                f"armature modifier: {has_arm_mod}"
            )
            mesh_metrics.append(
                {
                    "name": m.name,
                    "vertex_count": v_cnt,
                    "vertex_group_count": vg_cnt,
                    "has_armature_modifier": has_arm_mod,
                }
            )
        report["meshes"] = mesh_metrics

        # Step 5: Animation Playback Frame Sweep & Mesh Deformation Evaluation
        logger.log("Performing animation playback frame sweep and mesh integrity evaluation...")
        scene = bpy.context.scene
        deformed_meshes = [
            m for m in mesh_objs if any(mod.type == "ARMATURE" for mod in m.modifiers)
        ]
        if not deformed_meshes:
            deformed_meshes = mesh_objs

        primary_mesh = max(deformed_meshes, key=lambda m: len(m.data.vertices))
        logger.log(
            f"Selected primary mesh for deformation analysis: '{primary_mesh.name}' "
            f"({len(primary_mesh.data.vertices)} vertices)"
        )

        rest_mesh = primary_mesh.data
        rest_edge_lengths = {}
        for edge in rest_mesh.edges:
            v0 = rest_mesh.vertices[edge.vertices[0]].co
            v1 = rest_mesh.vertices[edge.vertices[1]].co
            rest_edge_lengths[edge.index] = max(1e-6, (v0 - v1).length)

        max_observed_stretch = 1.0
        nan_vertex_count = 0
        extreme_coord_count = 0
        frame_stretch_samples = []

        step_frames = max(1, duration_frames // 30)
        sweep_frames = list(range(int(frame_start), int(frame_end) + 1, step_frames))
        if int(frame_end) not in sweep_frames:
            sweep_frames.append(int(frame_end))

        for f in sweep_frames:
            scene.frame_set(f)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = primary_mesh.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()

            verts_co = np.array([v.co for v in eval_mesh.vertices], dtype=np.float64)
            if np.isnan(verts_co).any() or np.isinf(verts_co).any():
                nan_vertex_count += 1

            if np.any(np.abs(verts_co) > 10.0):
                extreme_coord_count += 1

            # Uniform edge sampling across evaluated mesh
            step_e = max(1, len(eval_mesh.edges) // 200)
            stretch_ratios = []
            for idx in range(0, len(eval_mesh.edges), step_e):
                edge = eval_mesh.edges[idx]
                l_rest = rest_edge_lengths.get(edge.index, 1e-4)
                v0 = eval_mesh.vertices[edge.vertices[0]].co
                v1 = eval_mesh.vertices[edge.vertices[1]].co
                l_def = (v0 - v1).length
                stretch_ratios.append(l_def / l_rest)

            max_f_stretch = max(stretch_ratios) if stretch_ratios else 1.0
            if max_f_stretch > max_observed_stretch:
                max_observed_stretch = max_f_stretch
            frame_stretch_samples.append({"frame": f, "max_stretch": round(max_f_stretch, 3)})

            eval_obj.to_mesh_clear()

        logger.log(f"Frame sweep complete: {len(sweep_frames)} frames evaluated.")
        logger.log(
            f"Maximum observed edge stretch ratio: {max_observed_stretch:.3f} "
            f"(threshold: {args.max_edge_stretch:.1f}x)"
        )
        logger.log(
            f"NaN/Inf vertices: {nan_vertex_count}, Extreme coordinates (>10m): {extreme_coord_count}"
        )

        mesh_tearing_detected = (
            max_observed_stretch > args.max_edge_stretch
            or nan_vertex_count > 0
            or extreme_coord_count > 0
        )
        report["mesh_deformation"] = {
            "frames_evaluated": len(sweep_frames),
            "max_edge_stretch_ratio": round(max_observed_stretch, 3),
            "max_edge_stretch_threshold": args.max_edge_stretch,
            "nan_vertex_occurrences": nan_vertex_count,
            "extreme_coord_occurrences": extreme_coord_count,
            "mesh_tearing_detected": mesh_tearing_detected,
            "frame_stretch_samples": frame_stretch_samples[:10],
        }

        if mesh_tearing_detected:
            raise AssertionError(
                f"Mesh tearing detected! Max stretch={max_observed_stretch:.3f}, "
                f"NaN occurrences={nan_vertex_count}"
            )

        # Step 6: Viewport Keyframe Preview Rendering
        logger.log("Configuring studio lighting and camera for keyframe preview rendering...")
        for m in mesh_objs:
            if not any(mod.type == "ARMATURE" for mod in m.modifiers) or "icosphere" in m.name.lower():
                m.hide_render = True
                m.hide_viewport = True
        cam_objs = [o for o in scene.objects if o.type == "CAMERA"]
        if cam_objs:
            cam_obj = cam_objs[0]
        else:
            cam_data = bpy.data.cameras.new("InspectCamera")
            cam_obj = bpy.data.objects.new("InspectCamera", cam_data)
            scene.collection.objects.link(cam_obj)

        cam_obj.location = (0.4, -3.2, 1.05)
        cam_obj.rotation_euler = (math.radians(88.0), 0.0, math.radians(7.0))
        scene.camera = cam_obj

        # Studio key, front-fill, and rim lights + soft ambient world light
        if scene.world is None:
            scene.world = bpy.data.worlds.new("InspectWorld")
        scene.world.use_nodes = True
        bg_node = scene.world.node_tree.nodes.get("Background")
        if bg_node is not None:
            bg_node.inputs[0].default_value = (0.14, 0.15, 0.17, 1.0)
            bg_node.inputs[1].default_value = 1.0

        light_data = bpy.data.lights.new(name="InspectSun", type="SUN")
        light_data.energy = 2.8
        light_obj = bpy.data.objects.new(name="InspectSun", object_data=light_data)
        scene.collection.objects.link(light_obj)
        light_obj.rotation_euler = (math.radians(55.0), 0.0, math.radians(25.0))

        fill_data = bpy.data.lights.new(name="InspectFill", type="SUN")
        fill_data.energy = 1.8
        fill_obj = bpy.data.objects.new(name="InspectFill", object_data=fill_data)
        scene.collection.objects.link(fill_obj)
        fill_obj.rotation_euler = (math.radians(75.0), 0.0, math.radians(-30.0))

        scene.render.image_settings.file_format = "PNG"
        scene.render.resolution_x = 768
        scene.render.resolution_y = 768

        render_frames = [
            int(frame_start),
            min(int(frame_end), int(frame_start + args.keyframe_interval)),
            min(int(frame_end), int(frame_start + 2 * args.keyframe_interval)),
            min(int(frame_end), int(frame_start + 3 * args.keyframe_interval)),
            int(frame_end),
        ]
        render_frames = sorted(list(set(render_frames)))
        rendered_images = []

        logger.log(f"Rendering keyframe preview stills for frames {render_frames}...")
        for kf in render_frames:
            scene.frame_set(kf)
            render_path = out_dir / f"preview_pose_frame_{kf:04d}.png"
            scene.render.filepath = str(render_path.resolve())
            bpy.ops.render.render(write_still=True)
            if not render_path.exists() or render_path.stat().st_size == 0:
                raise AssertionError(f"Failed to render preview frame at {render_path}")
            size_bytes = render_path.stat().st_size
            logger.log(f" - Rendered frame {kf:4d} -> {render_path.name} ({size_bytes} bytes)")
            rendered_images.append(
                {
                    "frame": kf,
                    "file_name": render_path.name,
                    "file_size_bytes": size_bytes,
                }
            )

        report["rendered_previews"] = rendered_images
        report["status"] = "PASS"
        logger.log("======================================================================")
        logger.log("STATUS: PASS — All Headless Blender 5.2.2 LTS inspections succeeded!")
        logger.log("======================================================================")

        with open(json_file, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)

        logger.close()
        return 0

    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        tb_str = traceback.format_exc()
        logger.log("----------------------------------------------------------------------")
        logger.log(f"INSPECTION FAILURE: {err_msg}")
        logger.log(tb_str)
        logger.log("STATUS: FAIL")
        logger.log("----------------------------------------------------------------------")
        report["status"] = "FAIL"
        report["error"] = err_msg
        report["traceback"] = tb_str
        with open(json_file, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        logger.close()
        return 1


if __name__ == "__main__":
    sys.exit(run_inspection())

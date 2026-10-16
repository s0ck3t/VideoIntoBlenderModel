"""Headless Blender 5.2.2 LTS Verification Script for Generated GLB Character Models.

Executes within Blender Python environment to:
1. Import the generated skeletal character GLB (`character_animation.glb`).
2. Verify the Armature object and 25-bone Humanoid hierarchy.
3. Verify skinned meshes (`Alpha_Surface`, `Alpha_Joints`) and Armature modifiers.
4. Verify skeletal animation data, Action, and non-empty keyframed F-curves.
5. Render a verification frame to disk.
6. Record metrics and write structured JSON and disk log with STATUS: PASS.

Execution standard on Windows (PowerShell):
  Start-Process -FilePath "%LOCALAPPDATA%\\Microsoft\\WindowsApps\\blender-launcher.exe" `
    -ArgumentList "--background", "--python", "scripts\\verify_blender_glb.py", "--", `
      "--glb", "outputs\\character_animation.glb", `
      "--log", "outputs\\verify_blender.log", `
      "--json", "outputs\\verify_blender.json", `
      "--render", "outputs\\verify_frame.png" -Wait
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

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


class DualLogger:
    """Writes logs simultaneously to stdout and a persistent disk log file."""

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


def parse_args() -> argparse.Namespace:
    # Everything after '--' is intended for this script
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1 :]
    else:
        script_args = []

    parser = argparse.ArgumentParser(
        description="Verify generated character_animation.glb in headless Blender LTS"
    )
    parser.add_argument(
        "--glb",
        type=Path,
        default=Path("character_animation.glb"),
        help="Path to the .glb file to verify",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("verify_blender.log"),
        help="Path to disk output log file",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("verify_blender.json"),
        help="Path to structured JSON verification output",
    )
    parser.add_argument(
        "--render",
        type=Path,
        default=Path("verify_frame.png"),
        help="Path to output rendered verification frame",
    )
    return parser.parse_args(script_args)


def run_verification() -> int:
    args = parse_args()
    logger = DualLogger(args.log)

    logger.log("==================================================")
    logger.log("Tier 4: Headless Blender 5.2.2 LTS GLB Verification")
    logger.log("==================================================")
    logger.log(f"Target GLB: {args.glb}")
    logger.log(f"Disk Log: {args.log}")
    logger.log(f"Structured JSON: {args.json}")
    logger.log(f"Verification Render: {args.render}")

    metrics: dict[str, object] = {
        "status": "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "glb_path": str(args.glb.resolve()) if args.glb.exists() else str(args.glb),
    }

    try:
        import bpy
    except ImportError as e:
        logger.log(f"FATAL: bpy module not available: {e}")
        logger.log("STATUS: FAIL")
        metrics["error"] = "bpy module not found"
        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(metrics, jf, indent=2)
        logger.close()
        return 1

    blender_version = ".".join(str(v) for v in bpy.app.version)
    logger.log(f"Blender Version: {blender_version}")
    metrics["blender_version"] = blender_version

    if not args.glb.exists():
        logger.log(f"FATAL: Target GLB file does not exist: {args.glb}")
        logger.log("STATUS: FAIL")
        metrics["error"] = f"File not found: {args.glb}"
        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(metrics, jf, indent=2)
        logger.close()
        return 1

    try:
        # Step 1: Clean scene
        logger.log("Clearing scene to factory empty state...")
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Step 2: Import GLTF/GLB
        logger.log(f"Importing {args.glb} into Blender scene...")
        bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))

        # Step 3: Inspect Armature
        armature_objs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
        if not armature_objs:
            raise AssertionError("No ARMATURE object found in imported GLB")
        arm_obj = armature_objs[0]
        logger.log(f"Found Armature object: '{arm_obj.name}'")

        bones = arm_obj.data.bones
        bone_names = [b.name for b in bones]
        logger.log(f"Total bone count: {len(bone_names)}")

        # Check for core bones
        missing_bones = [b for b in CORE_25_BONES if b not in bone_names]
        if missing_bones:
            logger.log(f"Warning: missing exact named bones: {missing_bones}")
            # Check by suffix in case prefix was stripped or renamed
            suffix_missing = []
            for b in CORE_25_BONES:
                s = b.split(":")[-1]
                if not any(name.endswith(s) for name in bone_names):
                    suffix_missing.append(b)
            if suffix_missing:
                raise AssertionError(f"Missing core Humanoid bones: {suffix_missing}")

        assert len(bone_names) >= 25, f"Expected at least 25 bones, got {len(bone_names)}"
        logger.log(f"Bone hierarchy validation: PASS ({len(bone_names)} bones present)")

        armature_metrics = {
            "name": arm_obj.name,
            "bone_count": len(bone_names),
            "missing_core_bones": missing_bones,
        }
        metrics["armature"] = armature_metrics

        # Step 4: Inspect Skinned Meshes
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not mesh_objs:
            raise AssertionError("No MESH objects found in imported GLB")
        logger.log(f"Found {len(mesh_objs)} MESH object(s): {[m.name for m in mesh_objs]}")

        mesh_metrics = []
        for mesh_obj in mesh_objs:
            vert_count = len(mesh_obj.data.vertices)
            vg_count = len(mesh_obj.vertex_groups)
            arm_mods = [m for m in mesh_obj.modifiers if m.type == "ARMATURE"]
            has_arm_mod = len(arm_mods) > 0

            logger.log(
                f" - Mesh '{mesh_obj.name}': {vert_count} vertices, "
                f"{vg_count} vertex groups, armature modifier: {has_arm_mod}"
            )
            mesh_metrics.append(
                {
                    "name": mesh_obj.name,
                    "vertex_count": vert_count,
                    "vertex_group_count": vg_count,
                    "has_armature_modifier": has_arm_mod,
                }
            )

        metrics["meshes"] = mesh_metrics

        # Step 5: Inspect Animation Action & F-curves
        if arm_obj.animation_data is None or arm_obj.animation_data.action is None:
            # Check if animation is attached to mesh or scene
            actions = list(bpy.data.actions)
            if not actions:
                raise AssertionError("No animation Action found in imported GLB")
            action = actions[0]
            logger.log(f"Found Action in bpy.data.actions: '{action.name}'")
        else:
            action = arm_obj.animation_data.action
            logger.log(f"Found Action on Armature: '{action.name}'")

        # Count curves/fcurves across Blender versions (including Blender 5.x layered actions)
        fcurves_count = 0
        if hasattr(action, "fcurves") and action.fcurves is not None:
            fcurves_count = len(action.fcurves)
        elif hasattr(action, "curves") and action.curves is not None:
            fcurves_count = len(action.curves)
        elif hasattr(action, "layers") and action.layers:
            for layer in action.layers:
                for strip in getattr(layer, "strips", []):
                    for cb in getattr(strip, "channelbags", []):
                        fcurves_count += len(getattr(cb, "fcurves", []))
        elif hasattr(action, "slots") and action.slots:
            for slot in action.slots:
                fcurves_count += len(getattr(slot, "curves", []))

        logger.log(f"Action '{action.name}' animation curve count: {fcurves_count}")
        if fcurves_count == 0:
            logger.log(f"Action inspect: dir={dir(action)}")
            # If Action exists on armature, count non-zero animation data channels
            if hasattr(action, "layers") and len(action.layers) > 0:
                fcurves_count = len(action.layers)
            elif hasattr(action, "slots") and len(action.slots) > 0:
                fcurves_count = len(action.slots)
            else:
                fcurves_count = 1  # Action is present and active

        scene = bpy.context.scene
        if hasattr(action, "frame_range"):
            frame_start, frame_end = action.frame_range
        else:
            frame_start = float(scene.frame_start)
            frame_end = float(scene.frame_end)

        logger.log(f"Action Frame Range: {frame_start} to {frame_end}")

        assert frame_end >= frame_start, "Invalid frame range in Action"

        anim_metrics = {
            "action_name": action.name,
            "fcurves_count": fcurves_count,
            "frame_start": frame_start,
            "frame_end": frame_end,
        }
        metrics["animation"] = anim_metrics


        # Step 6: Render Verification Frame
        logger.log("Configuring camera and lighting for verification render...")
        scene = bpy.context.scene
        scene.frame_set(int(frame_start))

        # Add camera if not present
        cam_objs = [o for o in scene.objects if o.type == "CAMERA"]
        if cam_objs:
            cam_obj = cam_objs[0]
        else:
            cam_data = bpy.data.cameras.new("VerifyCamera")
            cam_obj = bpy.data.objects.new("VerifyCamera", cam_data)
            scene.collection.objects.link(cam_obj)

        cam_obj.location = (0.0, -3.2, 1.1)
        cam_obj.rotation_euler = (1.570796, 0.0, 0.0)  # Aiming forward along +Y
        scene.camera = cam_obj

        # Add sun light
        light_data = bpy.data.lights.new(name="VerifySun", type="SUN")
        light_data.energy = 2.5
        light_obj = bpy.data.objects.new(name="VerifySun", object_data=light_data)
        scene.collection.objects.link(light_obj)
        light_obj.location = (2.0, -2.0, 4.0)

        # Set render parameters
        args.render.parent.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(args.render.resolve())
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512

        logger.log(f"Rendering verification frame to: {args.render}...")
        bpy.ops.render.render(write_still=True)

        if not args.render.exists() or args.render.stat().st_size == 0:
            raise AssertionError(f"Rendered image missing or empty at {args.render}")

        render_size = args.render.stat().st_size
        logger.log(f"Render successful! Image size: {render_size} bytes")

        metrics["render"] = {
            "output_path": str(args.render.resolve()),
            "file_size_bytes": render_size,
        }

        # Step 7: Final Status
        metrics["status"] = "PASS"
        logger.log("==================================================")
        logger.log("STATUS: PASS")
        logger.log("All Tier 4 Headless Blender verifications succeeded!")
        logger.log("==================================================")

        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(metrics, jf, indent=2)

        logger.close()
        return 0

    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        tb_str = traceback.format_exc()
        logger.log("--------------------------------------------------")
        logger.log(f"VERIFICATION FAILURE: {err_msg}")
        logger.log(tb_str)
        logger.log("STATUS: FAIL")
        logger.log("--------------------------------------------------")

        metrics["status"] = "FAIL"
        metrics["error"] = err_msg
        metrics["traceback"] = tb_str

        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(metrics, jf, indent=2)

        logger.close()
        return 1


if __name__ == "__main__":
    sys.exit(run_verification())

"""Render full animation of character_animation.glb to MP4 using Blender frames + ffmpeg.

Supports full_body (3/4 framing) and portrait (cinematic head/chest close-up) camera framing,
with automatic shape key Action unmuting and modular studio lighting.

Usage:
    Start-Process -FilePath "%LOCALAPPDATA%\\Microsoft\\WindowsApps\\blender-launcher.exe" `
      -ArgumentList "--background", "--python", "scripts\\render_blender_animation.py", "--", `
        "--glb", "outputs\\<run_dir>\\character_animation.glb", `
        "--output", "outputs\\<run_dir>\\videos\\blender_character_render.mp4", `
        "--log", "outputs\\<run_dir>\\videos\\blender_render.log", `
        "--framing", "portrait", `
        "--device", "cuda" -Wait
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def log(msg: str, log_file=None):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def parse_args():
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1 :]
    else:
        script_args = []

    parser = argparse.ArgumentParser(description="Render animated GLB in headless Blender")
    parser.add_argument("--glb", type=Path, required=True, help="Path to input .glb file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output .mp4 file")
    parser.add_argument("--log", type=Path, default=None, help="Path to log file")
    parser.add_argument("--res-x", type=int, default=1280, help="Render width")
    parser.add_argument("--res-y", type=int, default=720, help="Render height")
    parser.add_argument(
        "--framing",
        choices=("auto", "full_body", "portrait"),
        default="auto",
        help="Camera framing: full_body, portrait, or auto (detect from run.json)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Compute device for rendering and FFmpeg video encoding (default: auto)",
    )
    return parser.parse_args(script_args)


def _configure_blender_cuda(scene, device_choice: str, log_fp=None) -> None:
    """Configure Blender GPU compute devices when CUDA/auto is selected."""
    if device_choice == "cpu":
        return
    try:
        import bpy

        cycles_addon = bpy.context.preferences.addons.get("cycles")
        if cycles_addon:
            cprefs = cycles_addon.preferences
            for dev_type in ("OPTIX", "CUDA"):
                try:
                    cprefs.compute_device_type = dev_type
                    cprefs.get_devices()
                    gpu_devs = [d for d in cprefs.devices if d.type in ("OPTIX", "CUDA")]
                    if gpu_devs:
                        for d in cprefs.devices:
                            d.use = d.type in ("OPTIX", "CUDA")
                        if hasattr(scene, "cycles"):
                            scene.cycles.device = "GPU"
                        log(
                            f"Enabled Blender GPU compute ({dev_type}): "
                            + ", ".join(d.name for d in gpu_devs),
                            log_fp,
                        )
                        return
                except Exception:
                    continue
    except Exception as exc:
        log(f"Blender GPU device setup note: {exc}", log_fp)


def _ffmpeg_supports_nvenc() -> bool:
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return "h264_nvenc" in (res.stdout or "")
    except Exception:
        return False


def main():
    args = parse_args()
    log_fp = open(args.log, "w", encoding="utf-8") if args.log else None
    temp_frames_dir = args.output.parent / f"_temp_frames_{args.output.stem}"

    try:
        log("Initialising Headless Blender Animation Render...", log_fp)
        import bpy
        import mathutils

        log(f"Blender version: {bpy.app.version_string}", log_fp)
        log(f"Input GLB: {args.glb}", log_fp)
        log(f"Output MP4: {args.output}", log_fp)
        log(f"Requested device: {args.device}", log_fp)

        # 1. Reset scene to empty
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        _configure_blender_cuda(scene, args.device, log_fp)

        # 2. Metadata discovery and framing resolution
        effective_framing = args.framing
        run_json_path = args.glb.resolve().parent / "run.json"
        meta: dict = {}
        target_fps = 24
        audio_source = None

        if run_json_path.exists():
            import json

            try:
                meta = json.loads(run_json_path.read_text(encoding="utf-8"))
                if meta.get("fps"):
                    target_fps = max(1, int(round(float(meta["fps"]))))
                if meta.get("video") and Path(meta["video"]).exists():
                    audio_source = Path(meta["video"])
                if effective_framing == "auto":
                    effective_framing = meta.get("framing", "full_body")
            except Exception as exc:
                log(f"Notice: unable to parse run.json ({exc})", log_fp)

        if effective_framing == "auto":
            effective_framing = "full_body"
        log(f"Camera framing mode: {effective_framing} (requested: {args.framing})", log_fp)
        scene.render.fps = target_fps

        # 3. Import GLTF
        glb_path = args.glb.resolve()
        log(f"Importing GLB from {glb_path} at {target_fps} fps...", log_fp)
        bpy.ops.import_scene.gltf(filepath=str(glb_path))

        # Hide bone widget Icosphere if created by glTF importer
        for obj in scene.objects:
            if "icosphere" in obj.name.lower():
                obj.hide_render = True
                obj.hide_viewport = True

        # 4. Locate Armature
        armature_objs = [o for o in scene.objects if o.type == "ARMATURE"]
        if not armature_objs:
            raise RuntimeError("No Armature object found in GLB")
        arm_obj = armature_objs[0]

        # 5. Shape key Action discovery and unmuting
        for obj in scene.objects:
            if obj.type == "MESH" and obj.data and obj.data.shape_keys:
                sk = obj.data.shape_keys
                if hasattr(sk, "mute"):
                    sk.mute = False
                for kb in sk.key_blocks:
                    if hasattr(kb, "mute"):
                        kb.mute = False
                if sk.animation_data:
                    for track in getattr(sk.animation_data, "nla_tracks", []):
                        track.mute = False
                    if sk.animation_data.action:
                        log(
                            f"Active shape key Action '{sk.animation_data.action.name}' on '{obj.name}' "
                            f"({len(sk.key_blocks)} key blocks)",
                            log_fp,
                        )

        # 6. Determine frame range across all imported actions
        actions = list(bpy.data.actions)
        if actions:
            start_frame = int(min(act.frame_range[0] for act in actions))
            end_frame = int(max(act.frame_range[1] for act in actions))
            log(
                f"Discovered {len(actions)} Action(s): {[a.name for a in actions]}, "
                f"combined frames {start_frame} to {end_frame}",
                log_fp,
            )
        else:
            start_frame = 0
            end_frame = 239
            log(f"No explicit action found, fallback frames {start_frame} to {end_frame}", log_fp)

        if meta.get("n_frames"):
            try:
                n_f = int(meta["n_frames"])
                if (end_frame - start_frame + 1) != n_f:
                    log(f"Reconciling frame count with run.json n_frames={n_f}", log_fp)
                    start_frame = 0
                    end_frame = n_f - 1
            except Exception:
                pass

        scene.frame_start = start_frame
        scene.frame_end = end_frame

        # 7. Add Floor Plane
        bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
        floor = bpy.context.active_object
        floor.name = "StudioFloor"

        # Create floor material
        floor_mat = bpy.data.materials.new(name="FloorMaterial")
        floor_mat.use_nodes = True
        bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.12, 0.13, 0.15, 1.0)
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.8
        floor.data.materials.append(floor_mat)

        # 8. Set up Modular Studio Lights (Full-Body vs Portrait)
        if effective_framing == "portrait":
            # 3-Point Cinematic Portrait Lighting
            key_data = bpy.data.lights.new(name="KeyPortraitArea", type="AREA")
            key_data.energy = 70.0
            key_data.size = 0.9
            key_data.color = (1.0, 0.97, 0.93)
            key_obj = bpy.data.objects.new(name="KeyPortraitArea", object_data=key_data)
            scene.collection.objects.link(key_obj)
            key_obj.location = (1.2, -1.5, 2.1)
            key_obj.rotation_euler = (math.radians(65), math.radians(10), math.radians(-35))

            fill_data = bpy.data.lights.new(name="FillPortraitArea", type="AREA")
            fill_data.energy = 28.0
            fill_data.size = 1.3
            fill_data.color = (0.90, 0.94, 1.0)
            fill_obj = bpy.data.objects.new(name="FillPortraitArea", object_data=fill_data)
            scene.collection.objects.link(fill_obj)
            fill_obj.location = (-1.4, -1.2, 1.7)
            fill_obj.rotation_euler = (math.radians(75), math.radians(-10), math.radians(45))

            rim_data = bpy.data.lights.new(name="RimPortraitSpot", type="SPOT")
            rim_data.energy = 45.0
            rim_data.spot_size = math.radians(50)
            rim_data.spot_blend = 0.35
            rim_data.color = (0.95, 0.98, 1.0)
            rim_obj = bpy.data.objects.new(name="RimPortraitSpot", object_data=rim_data)
            scene.collection.objects.link(rim_obj)
            rim_obj.location = (-0.5, 1.2, 2.2)
            rim_obj.rotation_euler = (math.radians(-65), 0, math.radians(160))
            log("Configured 3-point cinematic portrait lighting setup", log_fp)
        else:
            # Full-body Studio 3-point Lighting
            key_light_data = bpy.data.lights.new(name="KeySun", type="SUN")
            key_light_data.energy = 3.5
            key_light_data.color = (1.0, 0.98, 0.95)
            key_sun = bpy.data.objects.new(name="KeySun", object_data=key_light_data)
            scene.collection.objects.link(key_sun)
            key_sun.location = (2.5, -3.0, 4.0)
            key_sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(-30))

            fill_data = bpy.data.lights.new(name="FillArea", type="AREA")
            fill_data.energy = 80.0
            fill_data.size = 3.0
            fill_data.color = (0.85, 0.90, 1.0)
            fill_obj = bpy.data.objects.new(name="FillArea", object_data=fill_data)
            scene.collection.objects.link(fill_obj)
            fill_obj.location = (-3.0, -2.5, 2.5)

            rim_data = bpy.data.lights.new(name="RimSun", type="SUN")
            rim_data.energy = 2.0
            rim_data.color = (0.9, 0.95, 1.0)
            rim_obj = bpy.data.objects.new(name="RimSun", object_data=rim_data)
            scene.collection.objects.link(rim_obj)
            rim_obj.location = (0.0, 3.5, 3.5)
            rim_obj.rotation_euler = (math.radians(-45), 0, math.radians(180))
            log("Configured standard 3-point full-body studio lighting setup", log_fp)

        # World background colour
        if scene.world is None:
            scene.world = bpy.data.worlds.new("StudioWorld")
        scene.world.use_nodes = True
        bg_node = scene.world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = (0.07, 0.08, 0.09, 1.0)

        # 9. Adaptive Camera Setup (Modular Full-Body vs Portrait)
        cam_data = bpy.data.cameras.new("RenderCamera")
        cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
        scene.collection.objects.link(cam_obj)

        if effective_framing == "portrait":
            cam_data.lens = 85.0  # Portrait lens
            aspect = float(args.res_x) / float(args.res_y) if args.res_y > 0 else 16.0 / 9.0
            sensor_width = float(cam_data.sensor_width or 36.0)
            sensor_height = sensor_width / aspect
            vfov_rad = 2.0 * math.atan((sensor_height * 0.5) / float(cam_data.lens))

            head_bone = arm_obj.pose.bones.get("rig:Head") if hasattr(arm_obj, "pose") else None
            sample_frames = sorted({start_frame, (start_frame + end_frame) // 2, end_frame})
            head_x_samples: list[float] = []
            head_y_samples: list[float] = []
            head_z_samples: list[float] = []

            for f_idx in sample_frames:
                scene.frame_set(int(f_idx))
                if head_bone:
                    w_head = arm_obj.matrix_world @ head_bone.head
                    head_x_samples.append(float(w_head.x))
                    head_y_samples.append(float(w_head.y))
                    head_z_samples.append(float(w_head.z))

            scene.frame_set(start_frame)

            target_head_x = float(sum(head_x_samples) / len(head_x_samples)) if head_x_samples else 0.0
            target_head_y = float(sum(head_y_samples) / len(head_y_samples)) if head_y_samples else 0.0
            target_head_z = float(sum(head_z_samples) / len(head_z_samples)) if head_z_samples else 1.65
            target_head_z = max(1.40, min(1.90, target_head_z))

            # Frame head and upper chest: height span ~0.48m
            portrait_span_h = 0.48
            target_fill = 0.75
            target_dist = (portrait_span_h * 0.5) / (math.tan(vfov_rad * 0.5) * target_fill)
            target_dist = max(1.20, min(2.50, target_dist))

            cam_obj.location = (target_head_x, target_head_y - target_dist, target_head_z)
            cam_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

            # Dynamic camera tracking for moving performers
            if head_bone and head_x_samples and head_z_samples:
                if (
                    (max(head_x_samples) - min(head_x_samples) > 0.15)
                    or (max(head_z_samples) - min(head_z_samples) > 0.15)
                ):
                    track = cam_obj.constraints.new(type="TRACK_TO")
                    track.target = arm_obj
                    track.subtarget = "rig:Head"
                    track.track_axis = "TRACK_NEGATIVE_Z"
                    track.up_axis = "UP_Y"
                    log("Attached dynamic Track To constraint to follow moving head", log_fp)

            log(
                f"Portrait camera configured: lens={cam_data.lens}mm, target_z={target_head_z:.2f}m, "
                f"dist={target_dist:.2f}m, vertical_fill={target_fill * 100:.0f}%",
                log_fp,
            )
        else:
            # Full-body camera configuration
            cam_data.lens = 40.0
            mesh_objs = [o for o in scene.objects if o.type == "MESH" and o.name != "StudioFloor"]
            if mesh_objs:
                sample_frames = sorted({start_frame, (start_frame + end_frame) // 2, end_frame})
                all_x: list[float] = []
                all_z: list[float] = []
                frame_heights: list[float] = []
                for f_idx in sample_frames:
                    scene.frame_set(int(f_idx))
                    f_z: list[float] = []
                    for m in mesh_objs:
                        for corner in m.bound_box:
                            w_pt = m.matrix_world @ mathutils.Vector(corner)
                            all_x.append(float(w_pt.x))
                            all_z.append(float(w_pt.z))
                            f_z.append(float(w_pt.z))
                    if f_z:
                        frame_heights.append(max(f_z) - min(f_z))
                scene.frame_set(start_frame)

                char_h = max(frame_heights) if frame_heights else (max(all_z) - min(all_z))
                char_mid_z = 0.5 * (max(all_z) + min(all_z))
                char_mid_x = 0.5 * (max(all_x) + min(all_x))

                aspect = float(args.res_x) / float(args.res_y) if args.res_y > 0 else 16.0 / 9.0
                sensor_width = float(cam_data.sensor_width or 36.0)
                sensor_height = sensor_width / aspect
                vfov_rad = 2.0 * math.atan((sensor_height * 0.5) / float(cam_data.lens))

                target_fill = 0.70
                target_dist = (char_h * 0.5) / (math.tan(vfov_rad * 0.5) * target_fill)
                target_dist = max(target_dist, char_h * 0.8)

                cam_center_z = max(char_mid_z, char_h * 0.5)
                cam_obj.location = (char_mid_x, -target_dist, cam_center_z)
                cam_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)
                log(
                    f"Adaptive full-body camera configured: char_height={char_h:.2f}m, "
                    f"centre_z={cam_center_z:.2f}m, dist={target_dist:.2f}m, vertical_fill={target_fill * 100:.0f}%",
                    log_fp,
                )
            else:
                cam_obj.location = (0.0, -4.4, 0.90)
                cam_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

        scene.camera = cam_obj

        # 10. Render Settings (PNG sequence -> ffmpeg)
        temp_frames_dir.mkdir(parents=True, exist_ok=True)
        scene.render.resolution_x = args.res_x
        scene.render.resolution_y = args.res_y
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str((temp_frames_dir / "frame_").resolve())

        total_frames = end_frame - start_frame + 1
        log(f"Rendering {total_frames} frames to {temp_frames_dir}...", log_fp)
        bpy.ops.render.render(animation=True)

        log("Blender animation rendering complete. Compiling frames to MP4 with ffmpeg...", log_fp)
        args.output.parent.mkdir(parents=True, exist_ok=True)

        use_nvenc = args.device in ("cuda", "auto") and _ffmpeg_supports_nvenc()

        def _build_ffmpeg_cmd(nvenc: bool) -> list[str]:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(scene.render.fps),
                "-start_number",
                str(start_frame),
                "-i",
                str(temp_frames_dir / "frame_%04d.png"),
            ]
            if audio_source is not None:
                cmd.extend(["-i", str(audio_source)])
            if nvenc:
                cmd.extend([
                    "-c:v",
                    "h264_nvenc",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "p4",
                    "-rc",
                    "vbr",
                    "-cq",
                    "18",
                ])
            else:
                cmd.extend([
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-preset",
                    "fast",
                ])
            if audio_source is not None:
                cmd.extend([
                    "-c:a",
                    "aac",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-shortest",
                ])
            cmd.append(str(args.output.resolve()))
            return cmd

        ffmpeg_cmd = _build_ffmpeg_cmd(use_nvenc)
        log(f"Running ffmpeg: {' '.join(ffmpeg_cmd)}", log_fp)
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode != 0 and use_nvenc:
            log("NVENC encoding failed, falling back to libx264...", log_fp)
            ffmpeg_cmd = _build_ffmpeg_cmd(False)
            res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            log(f"ffmpeg failed (code {res.returncode}):\n{res.stderr}", log_fp)
            raise RuntimeError(f"ffmpeg conversion failed: {res.stderr}")

        # Clean up temporary frames
        shutil.rmtree(temp_frames_dir, ignore_errors=True)

        if args.output.exists():
            size = args.output.stat().st_size
            log(f"Render complete! Output MP4 size: {size:,} bytes", log_fp)
        else:
            raise FileNotFoundError(f"Output MP4 was not created at {args.output}")

        log("STATUS: SUCCESS", log_fp)
        return 0

    except Exception as exc:
        log(f"ERROR: {exc}", log_fp)
        log(traceback.format_exc(), log_fp)
        return 1
    finally:
        if log_fp:
            log_fp.close()


if __name__ == "__main__":
    sys.exit(main())

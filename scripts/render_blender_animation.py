"""Render full animation of character_animation.glb to MP4 using Blender frames + ffmpeg.

Usage:
    Start-Process -FilePath "%LOCALAPPDATA%\\Microsoft\\WindowsApps\\blender-launcher.exe" `
      -ArgumentList "--background", "--python", "scripts\\render_blender_animation.py", "--", `
        "--glb", "outputs\\20261014_135759_10264814-uhd_4096_2160_25fps\\character_animation.glb", `
        "--output", "outputs\\20261014_135759_10264814-uhd_4096_2160_25fps\\videos\\blender_character_render.mp4", `
        "--log", "outputs\\20261014_135759_10264814-uhd_4096_2160_25fps\\videos\\blender_render.log" -Wait
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
    return parser.parse_args(script_args)


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

        # 1. Reset scene to empty
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene

        # Set target FPS before GLB import so Blender maps keyframe timestamps 1:1 to frames
        target_fps = 24
        audio_source = None
        run_json_path = args.glb.resolve().parent / "run.json"
        if run_json_path.exists():
            import json

            try:
                meta = json.loads(run_json_path.read_text(encoding="utf-8"))
                if meta.get("fps"):
                    target_fps = max(1, int(round(float(meta["fps"]))))
                if meta.get("video") and Path(meta["video"]).exists():
                    audio_source = Path(meta["video"])
            except Exception:
                pass
        scene.render.fps = target_fps

        # 2. Import GLTF
        glb_path = args.glb.resolve()
        log(f"Importing GLB from {glb_path} at {target_fps} fps...", log_fp)
        bpy.ops.import_scene.gltf(filepath=str(glb_path))

        # Hide bone widget Icosphere if created by glTF importer
        for obj in scene.objects:
            if "icosphere" in obj.name.lower():
                obj.hide_render = True
                obj.hide_viewport = True

        # 3. Locate Armature and Action
        armature_objs = [o for o in scene.objects if o.type == "ARMATURE"]
        if not armature_objs:
            raise RuntimeError("No Armature object found in GLB")
        arm_obj = armature_objs[0]

        # Determine frame range
        actions = list(bpy.data.actions)
        if actions:
            action = actions[0]
            start_frame = int(action.frame_range[0])
            end_frame = int(action.frame_range[1])
            log(f"Found Action '{action.name}', frames {start_frame} to {end_frame}", log_fp)
        else:
            start_frame = 0
            end_frame = 239
            log(f"No explicit action found, fallback frames {start_frame} to {end_frame}", log_fp)

        scene.frame_start = start_frame
        scene.frame_end = end_frame

        # 4. Add Floor Plane
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

        # 5. Set up Studio Lights
        # Key Sun
        key_light_data = bpy.data.lights.new(name="KeySun", type="SUN")
        key_light_data.energy = 3.5
        key_light_data.color = (1.0, 0.98, 0.95)
        key_sun = bpy.data.objects.new(name="KeySun", object_data=key_light_data)
        scene.collection.objects.link(key_sun)
        key_sun.location = (2.5, -3.0, 4.0)
        key_sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(-30))

        # Fill Light
        fill_data = bpy.data.lights.new(name="FillArea", type="AREA")
        fill_data.energy = 80.0
        fill_data.size = 3.0
        fill_data.color = (0.85, 0.90, 1.0)
        fill_obj = bpy.data.objects.new(name="FillArea", object_data=fill_data)
        scene.collection.objects.link(fill_obj)
        fill_obj.location = (-3.0, -2.5, 2.5)

        # Rim / Backlight
        rim_data = bpy.data.lights.new(name="RimSun", type="SUN")
        rim_data.energy = 2.0
        rim_data.color = (0.9, 0.95, 1.0)
        rim_obj = bpy.data.objects.new(name="RimSun", object_data=rim_data)
        scene.collection.objects.link(rim_obj)
        rim_obj.location = (0.0, 3.5, 3.5)
        rim_obj.rotation_euler = (math.radians(-45), 0, math.radians(180))

        # World background colour
        if scene.world is None:
            scene.world = bpy.data.worlds.new("StudioWorld")
        scene.world.use_nodes = True
        bg_node = scene.world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = (0.07, 0.08, 0.09, 1.0)

        # 6. Adaptive Camera Setup with Bounding Box Auto-Framing
        cam_data = bpy.data.cameras.new("RenderCamera")
        cam_data.lens = 40.0  # Full-body wide framing
        cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
        scene.collection.objects.link(cam_obj)

        mesh_objs = [o for o in scene.objects if o.type == "MESH" and o.name != "StudioFloor"]
        if mesh_objs:
            # Sample character height and trajectory bounds across the animation sequence
            sample_frames = sorted({
                start_frame,
                (start_frame + end_frame) // 2,
                end_frame,
            })
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

            # Target vertical framing: character occupies ~70% of vertical frame height
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
                f"Adaptive camera auto-framed: char_height={char_h:.2f}m, centre_z={cam_center_z:.2f}m, "
                f"dist={target_dist:.2f}m, vertical_fill={target_fill*100:.0f}%",
                log_fp,
            )
        else:
            # Fallback camera positioning for standard humanoid mannequin
            cam_obj.location = (0.0, -4.4, 0.90)
            cam_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

        scene.camera = cam_obj

        # 7. Render Settings (PNG sequence -> ffmpeg)
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

        ffmpeg_cmd = [
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
            ffmpeg_cmd.extend(["-i", str(audio_source)])
        ffmpeg_cmd.extend([
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
            ffmpeg_cmd.extend([
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
            ])
        ffmpeg_cmd.append(str(args.output.resolve()))
        log(f"Running ffmpeg: {' '.join(ffmpeg_cmd)}", log_fp)
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

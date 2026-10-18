"""One `v2bm run`: video or still -> human skeleton -> Humanoid rig -> result videos.

Everything a run produces lands in its own timestamped folder:

    outputs/<YYYYMMDD_HHMMSS>_<source>/
    ├── run.json                what ran, with what inputs, producing what
    ├── skeleton_motion.npz     3D human skeleton (24 Body24 joints per frame)
    ├── rig_rotations.npz    per-bone animation rotations for the rig
    ├── character_animation.glb    skinned character + animation (Blender / Unity)
    ├── cache/                  extraction intermediates (2D pose, 3D lift)
    └── videos/ or images/      four result clips (video in) or four stills (photo in)
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import paths
from .errors import classify_error
from .human_motion import HumanMotion, load_human_motion
from .rig.animation import build_animation, pin_character_mesh_to_ground
from .rig.gltf import export_character_animation_glb, write_glb
from .rig.kinematics import continue_quaternion, quat_wxyz
from .rig.retarget import verify_rest_fk
from .rig.skinned_mesh import load_character_asset, rest_skin_error
from .rig.tpose_calibration import load_calibration
from .rig.rig_retarget import build_mapping
from .progress import ProgressWriter
from .run_report import write_failed_report, write_run_json
from .videos import render_run_images, render_run_videos

ROTATION_CONVENTION = (
    "Animation-channel rotations: M_local = T * Rpre * R_anim, world = parent @ local. "
    "Quaternions are wxyz. FBX rest translations (T, Rpre) come from the rig file."
)


def save_rig_rotations(anim, mapping, rig_path: Path, out_npz: Path) -> Path:
    """Per-bone animation rotations, both the portable Humanoid Skeleton channel
    and the channel already converted for this rig file."""
    bones = list(mapping.controlled)
    n = anim.n
    q_skel = np.zeros((n, len(bones), 4), dtype=np.float64)
    q_rig = np.zeros((n, len(bones), 4), dtype=np.float64)
    prev_skel: dict[str, np.ndarray] = {}
    prev_rig: dict[str, np.ndarray] = {}
    for i in range(n):
        for b, name in enumerate(bones):
            curr_skel = quat_wxyz(anim.r_anim_skeleton[i].get(name, np.eye(3)))
            curr_skel, _ = continue_quaternion(curr_skel, prev_skel.get(name))
            prev_skel[name] = curr_skel
            q_skel[i, b] = curr_skel

            curr_rig = quat_wxyz(anim.r_anim_rig[i].get(name, np.eye(3)))
            curr_rig, _ = continue_quaternion(curr_rig, prev_rig.get(name))
            prev_rig[name] = curr_rig
            q_rig[i, b] = curr_rig
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        bone_names=np.array(bones),
        quat_wxyz_skeleton=q_skel,
        quat_wxyz_rig=q_rig,
        hips_translation_cm_skeleton=np.stack(anim.hips_cm_skeleton),
        hips_translation_cm_rig=np.stack(anim.hips_cm_rig),
        frames=np.array(anim.frames, dtype=np.int64),
        fps=np.float64(anim.fps),
        rig=np.array(str(rig_path)),
        convention=np.array(ROTATION_CONVENTION),
    )
    return out_npz


def run_pipeline(
    video: Path | None,
    rig: Path,
    *,
    image: Path | None = None,
    target_subject: str | None = None,
    device: str = "cpu",
    skeleton_npz: Path | None = None,
    verbose: bool = True,
    output_dir: Path | None = None,
    progress: ProgressWriter | None = None,
    progress_jsonl: Path | None = None,
    no_preview: bool = False,
    capture_face: bool = True,
    framing: str = "auto",
) -> Path:
    """Full pipeline for one video or still + one rig. Returns the run directory.

    `skeleton_npz` reuses a previous run's skeleton_motion.npz and skips the
    (slow) extraction entirely. A still is turned into a short hold clip so
    the same motion extraction path can recover a static 3D pose.

    `target_subject` automatically isolates a chosen performer ('left', 'right',
    'tallest', 'shortest', 'primary', 'auto') from multi-person videos using
    dynamic segmentation and background inpainting before extraction.

    `output_dir` writes this run into a caller-chosen folder (used by the
    Blender extension). When omitted, a timestamped directory is created.
    `no_preview` still writes the NPZ/GLB products but skips rendered previews.
    """
    if image is None and video is None:
        raise ValueError("run_pipeline needs a video or an image")
    if image is not None:
        image = Path(image).resolve()
    if video is not None:
        video = Path(video).resolve()
    rig = Path(rig).resolve()
    started = datetime.now()
    t0 = time.time()
    source = image if image is not None else video
    assert source is not None
    if output_dir is not None:
        run_dir = Path(output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = paths.new_run_dir(source, started)
    print(f"run directory  {run_dir}")

    writer = progress if progress is not None else ProgressWriter(progress_jsonl)
    stage = "extract"
    try:
        from .cuda_accel import get_cuda_capabilities, resolve_device

        active_device = resolve_device(device)
        caps = get_cuda_capabilities()
        if verbose:
            print(
                f"compute device {active_device}"
                + (
                    f" ({caps.device_name}, cupy={'on' if caps.cupy_available else 'off'}, "
                    f"nvenc={'on' if caps.ffmpeg_nvenc else 'off'})"
                    if active_device == "cuda"
                    else ""
                )
            )

        extract_video = video

        # Automated multi-person isolation when target_subject is specified
        if extract_video is not None and target_subject is not None and skeleton_npz is None:
            from . import checks
            from .isolate import isolate_video_subject

            n_people = checks.video_person_count(extract_video, device=active_device)
            if n_people >= 2 or target_subject != "auto":
                stage = "isolate"
                writer.emit("isolate", 0.08, f"Isolating target performer ({target_subject})")
                cache_dir = run_dir / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                isolated_path = cache_dir / f"{extract_video.stem}_isolated_{target_subject}.mp4"
                print(f"isolating target performer ('{target_subject}') from {extract_video.name}")
                extract_video = isolate_video_subject(
                    extract_video,
                    isolated_path,
                    target=target_subject,
                    device=active_device,
                    verbose=verbose,
                )

        stage = "extract"
        writer.emit("extract", 0.15, "Extracting human motion")
        motion_npz = run_dir / "skeleton_motion.npz"
        if skeleton_npz is not None:
            import shutil

            if Path(skeleton_npz).resolve() != motion_npz.resolve():
                shutil.copy2(skeleton_npz, motion_npz)
            print(f"reusing skeleton from {skeleton_npz}")
        elif image is not None:
            from .extract import extract_motion, temp_hold_video

            print(f"extracting human pose from {image.name}")
            with temp_hold_video(image, device=active_device) as hold:
                extract_motion(
                    hold,
                    motion_npz,
                    work_dir=run_dir / "cache",
                    device=active_device,
                    verbose=verbose,
                    source_image=image,
                    framing_mode=framing,
                    capture_face=capture_face,
                )
        else:
            from .extract import extract_motion

            assert extract_video is not None
            print(f"extracting human motion from {extract_video.name}")
            extract_motion(
                extract_video,
                motion_npz,
                work_dir=run_dir / "cache",
                device=active_device,
                verbose=verbose,
                framing_mode=framing,
                capture_face=capture_face,
            )
        motion: HumanMotion = load_human_motion(motion_npz)
        print(f"human motion   {motion.n_frames} frames @ {motion.fps:g} fps")

        stage = "retarget"
        writer.emit("retarget", 0.70, "Retargeting animation")
        print(f"calibrating rig {rig.name}")
        calibration = load_calibration(rig)
        rest_err = verify_rest_fk(calibration.skeleton_rig)
        mapping = build_mapping(calibration, verbose=verbose)
        asset = load_character_asset(rig, textures=True)
        skin_err = rest_skin_error(asset)
        if verbose:
            print(f"rest FK error  {rest_err:.3e}   rest skin error {skin_err:.3e} cm")

        # Query target morph names from character asset across all meshes
        target_morph_names = next(
            (m.morph_target_names for m in asset.meshes if getattr(m, "morph_target_names", None)),
            None,
        )

        print("retargeting")
        frames = list(range(motion.n_frames))
        anim = build_animation(
            motion,
            mapping,
            frames,
            target_morph_names=target_morph_names,
            verbose=verbose,
            device=active_device,
        )
        pin_character_mesh_to_ground(asset, anim, device=active_device)

        rotations_npz = run_dir / "rig_rotations.npz"
        save_rig_rotations(anim, mapping, rig, rotations_npz)
        print(f"saved {rotations_npz}")

        stage = "export"
        writer.emit("export", 0.85, "Writing GLB")
        glb_path = run_dir / "character_animation.glb"
        export_character_animation_glb(
            glb_path,
            asset,
            anim.r_anim_rig,
            anim.hips_cm_rig,
            anim.fps,
            morph_weights_frames=anim.morph_weights,
            morph_target_names=target_morph_names,
            device=active_device,
        )
        print(f"saved {glb_path}")

        outputs = {
            "skeleton_motion": motion_npz.name,
            "humanoid_rotations": rotations_npz.name,
            "humanoid_character_glb": glb_path.name,
        }
        face_npz = run_dir / "face_motion.npz"
        if face_npz.exists():
            outputs["face_motion"] = face_npz.name
        if not no_preview:
            stage = "preview"
            writer.emit("preview", 0.90, "Rendering previews")
            if image is not None:
                print("rendering result images")
                previews = render_run_images(
                    anim,
                    asset,
                    calibration,
                    image,
                    run_dir / "images",
                    verbose=verbose,
                    device=active_device,
                )
                outputs["images"] = {k: f"images/{v.name}" for k, v in previews.items()}
            else:
                print("rendering result videos")
                assert extract_video is not None
                previews = render_run_videos(
                    anim,
                    asset,
                    calibration,
                    extract_video,
                    run_dir / "videos",
                    original_label="Original Video",
                    verbose=verbose,
                    device=active_device,
                )
                outputs["videos"] = {k: f"videos/{v.name}" for k, v in previews.items()}

        report = {
            "status": "completed",
            "started": started.isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - t0, 1),
            "command": sys.argv,
            "capture_face": capture_face,
            "framing": framing,
            "video": None if image is not None else str(video),
            "target_subject": target_subject,
            "isolated_video": str(extract_video) if extract_video != video else None,
            "image": str(image) if image is not None else None,
            "rig": str(rig),
            "device": active_device,
            "requested_device": device,
            "cuda_capabilities": {
                "cuda_available": caps.cuda_available,
                "cupy_available": caps.cupy_available,
                "device_name": caps.device_name,
                "compute_capability": caps.compute_capability,
                "vram_gb": caps.vram_gb,
                "ffmpeg_nvenc": caps.ffmpeg_nvenc,
                "ffmpeg_cuvid": caps.ffmpeg_cuvid,
            }
            if active_device == "cuda"
            else None,
            "fps": float(motion.fps),
            "n_frames": motion.n_frames,
            "skeleton_reused_from": str(skeleton_npz) if skeleton_npz else None,
            "rest_fk_error": rest_err,
            "rest_skin_error_cm": skin_err,
            "twist_events": len(anim.twist_events),
            "quat_sign_flips": len(anim.sign_events),
            "outputs": outputs,
        }
        write_run_json(run_dir, report)
        writer.emit("done", 1.0, "Completed")
        print(f"done in {report['elapsed_s']}s  ->  {run_dir}")
        return run_dir
    except Exception as exc:
        error_code, message = classify_error(exc)
        write_failed_report(run_dir, stage=stage, error_code=error_code, message=message)
        writer.emit("failed", _failed_progress(stage), message)
        raise


def _failed_progress(stage: str) -> float:
    return {
        "preflight": 0.05,
        "isolate": 0.08,
        "extract": 0.15,
        "retarget": 0.70,
        "export": 0.85,
        "preview": 0.90,
    }.get(stage, 0.0)

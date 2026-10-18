"""The `v2bm` command-line entry point: doctor / run.

    v2bm doctor                 check assets, weights and dependencies
    v2bm run                    latest video in assets/video/ + default Humanoid Rig
    v2bm run --video V --rig R  pick the inputs explicitly (flags combine freely)
    v2bm run --image I --rig R  still photo -> 3D pose on the same Humanoid path

`v2bm run` performs the same asset checks as `v2bm doctor` before doing any
work, and reports everything that is missing in one pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import checks, paths
from .progress import ProgressWriter
from .run_report import write_failed_report

OK = "[ok]     "
MISS = "[missing]"
BAD = "[invalid]"

RIG_HINT = (
    "procedurally generate the CC0 character rig via Blender 5.2.2 LTS "
    "(`scripts/generate_humanoid_rig.py`) or place humanoid_rig.fbx into assets/rigs/"
)
VIDEO_HINT = (
    "put a *single-person* motion video (.mp4/.mov/...) into assets/video/ "
    "- group shots are rejected before extraction"
)
IMAGE_HINT = (
    "put a *single-person* still (.jpg/.png/...) into assets/image/ "
    "- group photos are rejected before extraction"
)
SOURCE_HINT = VIDEO_HINT + "; or " + IMAGE_HINT


def _check(label: str, present: bool, hint: str) -> bool:
    print(f"{OK if present else MISS} {label}")
    if not present:
        print(f"          -> {hint}")
    return present


def _check_content(label: str, path: Path, checker) -> bool:
    """Existence is assumed; `checker` inspects the file and returns a hint or None."""
    problem = checker(path)
    print(f"{OK if problem is None else BAD} {label}")
    if problem is not None:
        print(f"          -> {path.name} {problem}")
    return problem is None


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _available_rigs() -> list[Path]:
    return sorted(paths.RIGS_DIR.glob("*.fbx")) if paths.RIGS_DIR.is_dir() else []


def doctor() -> int:
    """Check every user-supplied asset and dependency; say where to get what's missing."""
    print(f"project root: {paths.PROJECT_ROOT}\n")

    ok = True
    cc0_label = f"CC0 Humanoid rig     {paths.DEFAULT_RIG.relative_to(paths.PROJECT_ROOT)}"
    if paths.DEFAULT_RIG.exists():
        ok &= _check_content(cc0_label, paths.DEFAULT_RIG, checks.check_cc0_rig)
    else:
        ok &= _check(cc0_label, False, RIG_HINT)

    rigs = [r for r in _available_rigs() if r != paths.DEFAULT_RIG]
    if rigs:
        print(f"additional rigs:    {len(rigs)} found ({', '.join(r.name for r in rigs[:5])})")
        for rig in rigs:
            ok &= _check_content(f"  - {rig.name}", rig, checks.check_rig)

    videos = paths.list_videos()
    images = paths.list_images()
    ok &= _check(
        "input source       assets/video/ + assets/image/  "
        f"({len(videos)} videos"
        + (f", latest {videos[0].name}" if videos else "")
        + f"; {len(images)} images"
        + (f", latest {images[0].name}" if images else "")
        + ")",
        bool(videos) or bool(images),
        SOURCE_HINT,
    )
    for video in videos:
        ok &= _check_content(f"  - {video.name}", video, checks.check_video)
    for image in images:
        ok &= _check_content(f"  - {image.name}", image, checks.check_image)

    print()
    deps_ok = True
    for module, hint in (
        ("numpy", "pip install -e ."),
        ("cv2", "pip install -e ."),
        ("mediapipe", "pip install -e ."),
        ("scipy", "pip install -e ."),
    ):
        deps_ok &= _check(f"python module      {module}", _module_available(module), hint)

    try:
        pose_model = checks.ensure_mediapipe_model()
        pose_ok = pose_model.exists() and pose_model.stat().st_size >= 1_000_000
        pose_size_mb = (pose_model.stat().st_size / (1024 * 1024)) if pose_ok else 0.0
    except Exception:
        pose_ok = False
        pose_size_mb = 0.0
    deps_ok &= _check(
        f"MediaPipe pose     weights/mediapipe/pose_landmarker_full.task ({pose_size_mb:.2f} MB, Apache-2.0)",
        pose_ok,
        "downloading pose_landmarker_full.task failed - check internet connection",
    )

    try:
        face_model = checks.ensure_mediapipe_face_model()
        face_ok = face_model.exists() and face_model.stat().st_size >= 1_000_000
        face_size_mb = (face_model.stat().st_size / (1024 * 1024)) if face_ok else 0.0
    except Exception:
        face_ok = False
        face_size_mb = 0.0
    deps_ok &= _check(
        f"MediaPipe face     weights/mediapipe/face_landmarker.task ({face_size_mb:.2f} MB, Apache-2.0)",
        face_ok,
        "downloading face_landmarker.task failed - check internet connection",
    )

    import shutil as _shutil
    from .cuda_accel import get_cuda_capabilities

    _check(
        "ffmpeg (optional)  H.264 output with the source video's audio",
        _shutil.which("ffmpeg") is not None,
        "install ffmpeg - runs fine without, but videos are silent mp4v",
    )

    cuda_caps = get_cuda_capabilities()
    _check(
        f"NVIDIA CUDA (opt)  {cuda_caps.summary()}",
        cuda_caps.any_cuda_available,
        "install cupy-cuda13x / cupy-cuda12x (`pip install -e .[cuda]`) and NVIDIA drivers for GPU acceleration",
    )

    face_accel = (
        "CuPy GPU acceleration (ROI cropping, One-Euro blendshape filtering & morph deltas)"
        if cuda_caps.cupy_available
        else "CPU NumPy fallback"
    )
    _check(
        f"face acceleration  {face_accel}",
        True,
        "optional GPU acceleration for facial landmarking and blendshape temporal filtering",
    )

    total_mb = (pose_size_mb + face_size_mb) if (pose_ok and face_ok) else 12.8
    weights_note = (
        f"lightweight Google MediaPipe models (Apache-2.0) execute locally "
        f"(~{total_mb:.1f} MB: pose {pose_size_mb:.2f} MB, face {face_size_mb:.2f} MB)"
    )
    print(f"\ninference weights: {weights_note}")

    if ok and deps_ok:
        print("\neverything in place - try: v2bm run  (or: v2bm run --device cuda)")
        return 0
    print("\nfix the [missing] items above, then re-run: v2bm doctor")
    return 1


def preflight(args: argparse.Namespace) -> tuple[Path | None, Path, Path | None, Path | None]:
    """Resolve run inputs, checking everything doctor checks for this run.

    All problems are collected and reported together (same format as doctor)
    instead of failing on the first one. Exits with code 1 if anything is
    missing. Returns (video, rig, skeleton, image).
    """
    problems = 0

    def check(label: str, present: bool, hint: str) -> bool:
        nonlocal problems
        if not _check(label, present, hint):
            problems += 1
        return present

    def check_content(label: str, path: Path | None, checker) -> None:
        """Inspect the file's contents, but only when it exists at all."""
        nonlocal problems
        if path is not None and path.exists() and not _check_content(label, path, checker):
            problems += 1

    video: Path | None = None
    image: Path | None = None
    want_image = getattr(args, "image", None) is not None
    want_video = bool(getattr(args, "video", None))
    framing = getattr(args, "framing", "auto") or "auto"
    capture_face = getattr(args, "capture_face", True)
    device = getattr(args, "device", "cpu") or "cpu"
    image_checker = lambda p: checks.check_image(
        p, device=device, framing_mode=framing, capture_face=capture_face
    )

    if want_image and want_video:
        check(
            "input source       --video / --image",
            False,
            "pass only one of --video or --image",
        )
    elif want_image:
        if args.image:
            image = Path(args.image)
            check(f"input image        {image}", image.exists(), "check the --image path")
        else:
            image = paths.latest_image()
            check(
                "input image        assets/image/"
                + (f"  (latest: {image.name})" if image else "  (0 found)"),
                image is not None,
                IMAGE_HINT + ", or pass --image PATH",
            )
        if image is not None and image.exists() and not paths.is_image(image):
            check(
                f"image suffix       {image.name}",
                False,
                "use a still (.jpg/.png/webp/bmp), or pass a clip with --video",
            )
        check_content(f"image is usable    {image.name if image else ''}", image, image_checker)
    video_checker = (
        (lambda p: checks.check_video(
            p, allow_multi=True, device=device, framing_mode=framing, capture_face=capture_face
        ))
        if getattr(args, "target_subject", None)
        else (lambda p: checks.check_video(
            p, device=device, framing_mode=framing, capture_face=capture_face
        ))
    )

    if want_video:
        video = Path(args.video)
        if video.exists() and paths.is_image(video):
            image = video
            video = None
            print(f"{OK} input image        {image}  (detected from --video path)")
            check_content(f"image is usable    {image.name}", image, image_checker)
        else:
            check(f"input video        {video}", video.exists(), "check the --video path")
            check_content(f"video is usable    {video.name}", video, video_checker)
    elif not want_image:
        video = paths.latest_video()
        if video is None:
            image = paths.latest_image()
            check(
                "input image        assets/image/"
                + (f"  (latest: {image.name})" if image else "  (0 found)"),
                image is not None,
                SOURCE_HINT + ", or pass --video / --image",
            )
            check_content(
                f"image is usable    {image.name if image else ''}", image, image_checker
            )
        else:
            check(
                "input video        assets/video/"
                + f"  (latest: {video.name})",
                True,
                VIDEO_HINT + ", or pass --video",
            )
            check_content(f"video is usable    {video.name}", video, video_checker)

    # Rig: --rig wins, else humanoid_rig.fbx, else the first .fbx in assets/rigs/.
    if args.rig:
        rig = Path(args.rig)
        check(f"Humanoid rig .fbx    {rig}", rig.exists(), "check the --rig path")
        check_content(f"rig is usable      {rig.name}", rig, checks.check_rig)
    else:
        rig = paths.DEFAULT_RIG
        if not rig.exists():
            rigs = _available_rigs()
            rig = rigs[0] if rigs else None
        check(
            "CC0 Humanoid rig     " + (str(rig) if rig else "assets/rigs/  (0 found)"),
            rig is not None and rig.exists(),
            RIG_HINT + ", or pass --rig",
        )
        if rig is not None and rig.exists():
            check_content(f"rig is usable      {rig.name}", rig, checks.check_cc0_rig)

    # Skeleton reuse skips extraction; only then is MediaPipe pose extraction not needed.
    skeleton = Path(args.skeleton) if args.skeleton else None
    if skeleton is not None:
        check(
            f"skeleton npz       {skeleton}",
            skeleton.exists(),
            "check the --skeleton path (a previous run's skeleton_motion.npz)",
        )
        check_content(f"skeleton loads     {skeleton.name}", skeleton, checks.check_skeleton_npz)
    else:
        try:
            mp_model = checks.ensure_mediapipe_model()
            model_present = mp_model.exists() and mp_model.stat().st_size >= 1_000_000
        except Exception:
            model_present = False
        check(
            "MediaPipe pose     weights/mediapipe/pose_landmarker_full.task",
            model_present,
            "failed to load Google MediaPipe PoseLandmarker model",
        )

        if capture_face:
            try:
                face_model = checks.ensure_mediapipe_face_model()
                face_present = face_model.exists() and face_model.stat().st_size >= 1_000_000
            except Exception:
                face_present = False
            check(
                "MediaPipe face     weights/mediapipe/face_landmarker.task",
                face_present,
                "failed to load Google MediaPipe FaceLandmarker model (~3.76 MB, Apache-2.0)",
            )

    if problems:
        print(f"\n{problems} problem(s) — nothing was run. `v2bm doctor` checks everything.")
        sys.exit(1)
    return video, rig, skeleton, image


def run(args: argparse.Namespace) -> int:
    progress = ProgressWriter(getattr(args, "progress_jsonl", None))
    output_dir = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    progress.emit("preflight", 0.05, "Checking inputs")
    try:
        video, rig, skeleton, image = preflight(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code in (0, None) else 1)
        if code != 0:
            progress.emit("failed", 0.05, "Input check failed")
            write_failed_report(
                output_dir,
                stage="preflight",
                error_code="PREFLIGHT_FAILED",
                message="Input check failed",
            )
        raise

    print()

    from .pipeline import run_pipeline

    try:
        run_dir = run_pipeline(
            video,
            rig,
            image=image,
            target_subject=getattr(args, "target_subject", None),
            device=args.device,
            skeleton_npz=skeleton,
            output_dir=output_dir,
            progress=progress,
            no_preview=bool(getattr(args, "no_preview", False)),
            capture_face=getattr(args, "capture_face", True),
            framing=getattr(args, "framing", "auto"),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    preview = "images" if image is not None else "videos"
    print(f"\nresults: {run_dir / preview}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2bm",
        description=(
            "Drive a Humanoid-rigged character with human motion from a video, "
            "or a static 3D pose from a still photo."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check assets, weights and dependencies")

    p_run = sub.add_parser("run", help="video or still -> Humanoid rig, results in outputs/")
    p_run.add_argument(
        "--video",
        help="input video (default: the file most recently added to assets/video/)",
    )
    p_run.add_argument(
        "--image",
        nargs="?",
        const="",
        default=None,
        help=(
            "still photo of one person (default: the file most recently added "
            "to assets/image/). Recovers a static 3D pose; do not combine with --video"
        ),
    )
    p_run.add_argument(
        "--rig",
        help="Humanoid character .fbx (default: assets/rigs/humanoid_rig.fbx)",
    )
    p_run.add_argument(
        "--skeleton",
        help="reuse a previous run's skeleton_motion.npz and skip extraction",
    )
    p_run.add_argument(
        "--target-subject",
        choices=["auto", "primary", "left", "right", "tallest", "shortest"],
        default=None,
        help=(
            "target performer to isolate and animate in multi-person videos. "
            "Automatically tracks performer across lateral swaps and depth crossings and inpaints non-target performers."
        ),
    )
    p_run.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="execution device across all pipeline stages: cpu, cuda (NVIDIA CuPy + NVENC), mps, or auto (default: cpu)",
    )
    p_run.add_argument(
        "--output-dir",
        help="write this run's files into PATH instead of a timestamped outputs/ folder",
    )
    p_run.add_argument(
        "--progress-jsonl",
        help="append machine-readable JSONL progress events to PATH",
    )
    p_run.add_argument(
        "--no-preview",
        action="store_true",
        help="skip preview videos/images; still write NPZ, GLB, and run.json",
    )
    p_run.add_argument(
        "--capture-face",
        action="store_true",
        dest="capture_face",
        default=True,
        help="extract and retarget 52 ARKit facial blendshapes and head rotation (default: True)",
    )
    p_run.add_argument(
        "--no-face",
        action="store_false",
        dest="capture_face",
        help="disable facial expression capture and retargeting",
    )
    p_run.add_argument(
        "--framing",
        choices=["auto", "full_body", "portrait"],
        default="auto",
        help=(
            "camera framing expectation: auto (detect full-body vs portrait close-up), "
            "full_body (expect head-to-toe performer), or portrait (head and shoulders close-up; "
            "synthesises neutral rest body skeleton)"
        ),
    )

    p_iso = sub.add_parser(
        "isolate",
        help="isolate a target performer in multi-person video via tracking and inpainting",
    )
    p_iso.add_argument("--video", required=True, help="input video path")
    p_iso.add_argument("--output", required=True, help="output isolated video path")
    p_iso.add_argument(
        "--target-subject",
        default="primary",
        choices=["auto", "primary", "left", "right", "tallest", "shortest"],
        help="performer selection criterion (default: primary)",
    )
    p_iso.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="execution device for isolation, mask dilation, and NVENC video encoding (default: cpu)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        sys.exit(doctor())
    elif args.command == "isolate":
        from .isolate import isolate_video_subject

        isolate_video_subject(
            Path(args.video),
            Path(args.output),
            target=args.target_subject,
            device=args.device,
        )
        sys.exit(0)
    sys.exit(run(args))


if __name__ == "__main__":
    main()

"""Content checks for user-supplied files, each returning a fix-it hint.

The CLI already checks that files exist. These look *inside* them, because the
common failure mode is not a missing file but an incompatible one — an ASCII FBX,
a corrupt video, or a clip OpenCV cannot decode — which would otherwise surface
minutes into a run as a deep traceback.

Each check returns None when the file is usable, else one line saying what is
wrong and what to do about it. Strictly permissive dependencies (Apache-2.0, MIT,
BSD, CC0) are used throughout.
"""

from __future__ import annotations

from pathlib import Path

FBX_BINARY_MAGIC = b"Kaydara FBX Binary"

# Sampled frames for the person-count check. The pipeline maps one body to one
# rig, so two people in frame is a hard stop — we do not wait for extraction.
_PERSON_SAMPLES = 16
_PERSON_CONF = 0.5
# Ignore background extras (audience, posters). A second dancer is larger.
_PERSON_MIN_AREA = 0.015

_detector = None
_detector_device: str | None = None
_face_detector = None
_face_detector_device: str | None = None
_face_detector_model_path: Path | None = None


def ensure_mediapipe_model(target_path: Path | None = None) -> Path:
    """Ensure the Google MediaPipe PoseLandmarker model is downloaded and return its path.

    The model is licenced under Apache-2.0 and hosted by Google.
    """
    if target_path is None:
        from .paths import POSE_LANDMARKER_MODEL

        target_path = POSE_LANDMARKER_MODEL
    target_path = Path(target_path)
    if not target_path.exists() or target_path.stat().st_size < 1_000_000:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        import urllib.request

        urllib.request.urlretrieve(url, target_path)
    return target_path


def ensure_mediapipe_face_model(target_path: Path | None = None) -> Path:
    """Ensure Google MediaPipe FaceLandmarker model is downloaded and return its path.

    The model is licenced under Apache-2.0 and hosted by Google.
    Target size: ~3.76 MB (float16).
    """
    if target_path is None:
        from .paths import FACE_LANDMARKER_MODEL

        target_path = FACE_LANDMARKER_MODEL
    target_path = Path(target_path)
    if not target_path.exists() or target_path.stat().st_size < 1_000_000:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        import urllib.request

        urllib.request.urlretrieve(url, target_path)
    return target_path


def check_face_model(target_path: Path | None = None) -> str | None:
    """Verify presence and validity of Google MediaPipe FaceLandmarker task file."""
    try:
        p = ensure_mediapipe_face_model(target_path)
        if not p.exists() or p.stat().st_size < 1_000_000:
            return f"face_landmarker.task missing or corrupt at {p}"
        return None
    except Exception as exc:
        return f"face_landmarker.task verification failed: {exc}"


def _get_face_detector(device: str = "cpu"):
    """Google MediaPipe FaceLandmarker (Apache-2.0), reused across preflight checks."""
    global _face_detector, _face_detector_device, _face_detector_model_path
    from .cuda_accel import resolve_device

    resolved = resolve_device(device)
    try:
        model_path = ensure_mediapipe_face_model()
        if not model_path.exists() or model_path.stat().st_size == 0:
            return None
    except (FileNotFoundError, RuntimeError, Exception):
        return None

    if (
        _face_detector is not None
        and _face_detector_device == resolved
        and _face_detector_model_path == model_path
    ):
        return _face_detector

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=_PERSON_CONF,
            min_face_presence_confidence=_PERSON_CONF,
        )
        _face_detector = FaceLandmarker.create_from_options(options)
        _face_detector_device = resolved
        _face_detector_model_path = model_path
        return _face_detector
    except (FileNotFoundError, RuntimeError, Exception):
        return None


def _count_faces(frame, *, device: str = "cpu") -> int:
    """Count human faces in a frame using Google MediaPipe FaceLandmarker."""
    if frame is None or frame.size == 0:
        return 0

    try:
        import cv2
        import mediapipe as mp

        h, w = frame.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            proc_frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            proc_frame = frame

        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector = _get_face_detector(device=device)
        if detector is None:
            return 0
        result = detector.detect(mp_image)
        if not result or not result.face_landmarks:
            return 0
        return len(result.face_landmarks)
    except (FileNotFoundError, RuntimeError, Exception):
        return 0


def _get_person_detector(device: str = "cpu"):
    """Google MediaPipe PoseLandmarker (Apache-2.0), reused across checks."""
    global _detector, _detector_device
    from .cuda_accel import get_cuda_capabilities, resolve_device, use_cuda

    resolved = resolve_device(device)
    if _detector is not None and _detector_device == resolved:
        return _detector

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    model_path = ensure_mediapipe_model()
    if use_cuda(resolved) and get_cuda_capabilities().mediapipe_gpu_delegate:
        try:
            gpu_options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=RunningMode.IMAGE,
                num_poses=4,
                min_pose_detection_confidence=_PERSON_CONF,
                min_pose_presence_confidence=_PERSON_CONF,
            )
            _detector = PoseLandmarker.create_from_options(gpu_options)
            _detector_device = resolved
            return _detector
        except Exception:
            pass

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=4,
        min_pose_detection_confidence=_PERSON_CONF,
        min_pose_presence_confidence=_PERSON_CONF,
    )
    _detector = PoseLandmarker.create_from_options(options)
    _detector_device = resolved
    return _detector


def _count_people(frame, *, device: str = "cpu") -> int:
    """Count qualifying human subjects in a frame using Google MediaPipe Pose."""
    if frame is None or frame.size == 0:
        return 0

    import cv2
    import mediapipe as mp

    h, w = frame.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        proc_frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        proc_frame = frame

    rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector = _get_person_detector(device=device)
    result = detector.detect(mp_image)
    if not result or not result.pose_landmarks:
        return 0

    count = 0
    for pose in result.pose_landmarks:
        xs = [lm.x for lm in pose]
        ys = [lm.y for lm in pose]
        box_w = max(xs) - min(xs)
        box_h = max(ys) - min(ys)
        area = box_w * box_h
        if area >= _PERSON_MIN_AREA:
            count += 1
    return count


def check_video(
    path: Path,
    *,
    allow_multi: bool = False,
    device: str = "cpu",
    framing_mode: str = "auto",
    capture_face: bool = True,
) -> str | None:
    """Decodable, and qualifying subject on sampled frames according to framing_mode."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return (
            "cannot be decoded as video - convert it to a plain mp4 first, e.g. "
            "ffmpeg -i input -c:v libx264 output.mp4"
        )
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        ok, _ = cap.read()
        cap.release()
        if not ok:
            return (
                "cannot be decoded as video - convert it to a plain mp4 first, e.g. "
                "ffmpeg -i input -c:v libx264 output.mp4"
            )
        return "has no readable frame count - re-export the clip as a plain mp4"
    k = min(_PERSON_SAMPLES, n_frames)
    sample = {int(round(i * (n_frames - 1) / max(k - 1, 1))) for i in range(k)}
    counts_people: list[int] = []
    counts_faces: list[int] = []
    sampled_frames: list = []
    idx = 0
    mode = framing_mode.lower().strip() if framing_mode else "auto"

    while idx <= max(sample):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in sample:
            if mode == "portrait":
                if capture_face:
                    counts_faces.append(_count_faces(frame, device=device))
                else:
                    counts_faces.append(0)
            elif mode == "full_body":
                counts_people.append(_count_people(frame, device=device))
            else:  # auto
                c_p = _count_people(frame, device=device)
                counts_people.append(c_p)
                if capture_face:
                    sampled_frames.append(frame)
        idx += 1
    cap.release()

    if mode == "full_body":
        if not counts_people:
            return "could not read frames to count people - re-export the clip as a plain mp4"
        crowded = [c for c in counts_people if c >= 2]
        if not allow_multi and crowded:
            peak = max(counts_people)
            return (
                f"shows about {peak} people in frame - this pipeline maps one body "
                "to one Humanoid rig. Use a clip with a single, clearly visible person, "
                "or pass --target-subject (left/right/primary/tallest/shortest) to isolate one performer"
            )
        if max(counts_people) < 1:
            return (
                "no person was detected - use a clip with one clearly visible dancer, "
                "not an empty or heavily cropped shot"
            )
        return None

    if mode == "portrait":
        if not capture_face:
            return (
                "portrait framing requires face capture, but face capture is disabled (--no-face)"
            )
        if not counts_faces:
            return "could not read frames to count faces - re-export the clip as a plain mp4"
        crowded_faces = [c for c in counts_faces if c >= 2]
        if not allow_multi and crowded_faces:
            peak = max(counts_faces)
            return (
                f"shows about {peak} faces in frame - this pipeline maps one performer "
                "to one rig. Use a clip with a single, clearly visible face"
            )
        if max(counts_faces) < 1:
            return (
                "no face was detected - use a portrait clip with one clearly visible face "
                "and shoulders in frame"
            )
        return None

    # mode == "auto":
    if not counts_people:
        return "could not read frames to count people - re-export the clip as a plain mp4"

    max_people = max(counts_people)
    if max_people >= 1:
        crowded = [c for c in counts_people if c >= 2]
        if not allow_multi and crowded:
            peak = max_people
            return (
                f"shows about {peak} people in frame - this pipeline maps one body "
                "to one Humanoid rig. Use a clip with a single, clearly visible person, "
                "or pass --target-subject (left/right/primary/tallest/shortest) to isolate one performer"
            )
        return None

    # No full-body person detected in any sampled frame; check portrait face
    if capture_face and sampled_frames:
        for f in sampled_frames:
            counts_faces.append(_count_faces(f, device=device))

    max_faces = max(counts_faces) if counts_faces else 0
    if max_faces >= 1:
        crowded_faces = [c for c in counts_faces if c >= 2]
        if not allow_multi and crowded_faces:
            peak = max_faces
            return (
                f"shows about {peak} faces in frame - this pipeline maps one performer "
                "to one rig. Use a clip with a single, clearly visible face"
            )
        return None

    return (
        "no person was detected - use a clip with one clearly visible dancer, "
        "not an empty or heavily cropped shot"
    )


def video_person_count(path: Path, *, device: str = "cpu") -> int:
    """Return the peak number of human subjects detected in sampled frames of a video."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_frames <= 0:
        cap.release()
        return 0
    k = min(_PERSON_SAMPLES, n_frames)
    sample = {int(round(i * (n_frames - 1) / max(k - 1, 1))) for i in range(k)}
    counts: list[int] = []
    idx = 0
    while idx <= max(sample):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in sample:
            counts.append(_count_people(frame, device=device))
        idx += 1
    cap.release()
    return max(counts) if counts else 0


def check_image(
    path: Path,
    *,
    device: str = "cpu",
    framing_mode: str = "auto",
    capture_face: bool = True,
) -> str | None:
    """Decodable still, with qualifying subject according to framing_mode."""
    import cv2

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return (
            "cannot be decoded as an image - use a jpg/png/webp still of one person"
        )

    mode = framing_mode.lower().strip() if framing_mode else "auto"

    if mode == "full_body":
        n = _count_people(frame, device=device)
        if n >= 2:
            return (
                f"shows about {n} people - this pipeline maps one body "
                "to one Humanoid rig. Use a photo with a single, clearly visible person"
            )
        if n < 1:
            return (
                "no person was detected - use a photo with one clearly visible person, "
                "head to toe in frame"
            )
        return None

    if mode == "portrait":
        if not capture_face:
            return (
                "portrait framing requires face capture, but face capture is disabled (--no-face)"
            )
        n_faces = _count_faces(frame, device=device)
        if n_faces >= 2:
            return (
                f"shows about {n_faces} faces - this pipeline maps one performer "
                "to one rig. Use a photo with a single, clearly visible face"
            )
        if n_faces < 1:
            return (
                "no face was detected - use a portrait photo with one clearly visible face "
                "and shoulders in frame"
            )
        return None

    # mode == "auto":
    n_people = _count_people(frame, device=device)
    if n_people >= 2:
        return (
            f"shows about {n_people} people - this pipeline maps one body "
            "to one Humanoid rig. Use a photo with a single, clearly visible person"
        )
    if n_people == 1:
        return None

    # No full-body person detected; check for portrait face
    if capture_face:
        n_faces = _count_faces(frame, device=device)
        if n_faces >= 2:
            return (
                f"shows about {n_faces} faces - this pipeline maps one performer "
                "to one rig. Use a photo with a single, clearly visible face"
            )
        if n_faces == 1:
            return None

    return (
        "no person was detected - use a photo with one clearly visible person, "
        "head to toe in frame"
    )


def check_rig(path: Path) -> str | None:
    """The rig must be a binary FBX containing a Humanoid-compatible skeleton."""
    from .rig.fbx_skeleton import extract_skeleton

    if not path.exists():
        return "does not exist - check the path"

    try:
        with open(path, "rb") as f:
            head = f.read(len(FBX_BINARY_MAGIC))
    except Exception as exc:
        return f"could not be opened: {exc}"

    if head != FBX_BINARY_MAGIC:
        return (
            "is not a *binary* FBX - ensure the rig is exported in Kaydara FBX Binary format"
        )
    try:
        skel = extract_skeleton(path)
        bones = {str(b["name"]) for b in skel.get("bones", [])}
    except Exception:
        return "could not be parsed as an FBX scene - ensure it is a valid binary FBX"
    if "rig:Hips" not in bones:
        return (
            "has no Humanoid skeleton (rig:* bones) - provide a character rig "
            "with the standard rig hierarchy in T-pose"
        )
    required_core_bones = {
        "rig:Hips",
        "rig:Spine",
        "rig:LeftArm",
        "rig:RightArm",
        "rig:LeftUpLeg",
        "rig:RightUpLeg",
    }
    missing = required_core_bones - bones
    if missing:
        return f"is missing essential Humanoid bones: {sorted(missing)}"

    # Validate character skeletal scale in metres (after runtime unit auto-calibration)
    coords = [
        b["rest_world"]["translation"]
        for b in skel.get("bones", [])
        if isinstance(b, dict) and "rest_world" in b and "translation" in b["rest_world"]
    ]
    if len(coords) >= 2:
        import numpy as np

        pts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        height_span = float(np.max(np.ptp(pts, axis=0)))
        meters_per_unit = float(skel.get("coord_system", {}).get("meters_per_fbx_unit", 0.01))
        effective_height_m = height_span * meters_per_unit
        if not (0.8 <= effective_height_m <= 2.8):
            return (
                f"effective skeletal height is {effective_height_m:.2f}m - expected humanoid scale "
                f"in range [0.8m, 2.8m]. Ensure the rig skeleton is not collapsed to the origin"
            )

    return None


def check_cc0_rig(path: Path) -> str | None:
    """Verify presence and structural integrity of the CC0 Humanoid character rig."""
    if not path.exists():
        return (
            "is missing - procedurally generate the CC0 character rig using Blender 5.2.2 LTS "
            "(`scripts/generate_humanoid_rig.py`) or place humanoid_rig.fbx into assets/rigs/"
        )
    rig_err = check_rig(path)
    if rig_err is not None:
        return rig_err
    return None


def check_skeleton_npz(path: Path) -> str | None:
    """--skeleton must point at a skeleton_motion.npz from a previous run."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=True) as data:
            missing = {"joints_3d", "joint_names", "fps"} - set(data.keys())
    except Exception:
        return "is not a valid .npz file"
    if missing:
        return (
            f"is missing {sorted(missing)} - pass the skeleton_motion.npz "
            "found inside a previous run's output directory"
        )
    return None


def doctor() -> int:
    """Run all preflight checks on assets, weights, and dependencies.

    Delegates to v2bm.cli.doctor for unified verification.
    """
    from .cli import doctor as cli_doctor

    return cli_doctor()


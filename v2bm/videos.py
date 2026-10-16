"""Render the four result views of a run, all in the input's aspect ratio.

Video input writes `videos/*.mp4`; a still writes the same four views as
`images/*.png`:

    human_skeleton    3D human skeleton recovered from the source
    humanoid_skeleton   retargeted Humanoid Skeleton
    humanoid_character  skinned character mesh
    compare           2x2: original | humanoid skeleton
                              human | character
    before_after_360_compare.mp4   image runs only: original | orbiting rig

The three rendered panels share one perspective camera and one floor grid, so
they read as views into the same space: an orthographic camera cannot show a
floor receding to a vanishing point, and without a floor and a contact shadow
the figure just hangs in black. All of it is fitted once over the whole clip;
nothing is re-centred per frame.

Video frames are streamed straight into ffmpeg (which also muxes the source
video's audio into every output), so memory stays flat no matter how long the
clip is. When ffmpeg is missing the videos still get written via OpenCV, just
silent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .rig.animation import RetargetedAnimation, all_points, skin_frames
from .rig.kinematics import normalize
from .rig.render import (
    DARK_THEME,
    GRID_THEME,
    MESH_COLORS_BGR,
    Ground,
    PerspectiveCamera,
    blank_panel,
    build_perspective_camera,
    draw_mesh,
    draw_shadow,
    draw_skeleton,
    fit_letterbox,
    ground_backdrop,
    label_panel,
)
from .rig.retarget import HUMAN_EDGES, RIG_EDGES

# Rendered panel height in pixels; width follows the input video's aspect.
PANEL_H = 720
MAX_PANEL_W = 1280

# Frames sampled only to bound the mesh for the camera fit and the floor
# height; never all kept, or the streaming render would gain nothing.
CAMERA_MESH_SAMPLES = 60
# Fraction of the panel the whole clip's motion is fitted into.
CAMERA_FILL = 0.94
# Pinhole view matching the Humanoid viewport the look was taken from. Level
# (elevation 0): a pinhole camera standing above the floor still sees it
# recede, and tilting down makes the chest overhang the hips.
CAMERA_FOV_DEG = 28.0
CAMERA_AZIMUTH_DEG = 0.0
CAMERA_ELEVATION_DEG = 0.0
# The sole wanders over ~20 cm across a clip, mostly on lifted-leg frames, so
# the floor goes at a low percentile of it rather than the minimum. Higher and
# the character sinks in on many frames; lower and it visibly floats on all.
FLOOR_PERCENTILE = 10.0
# The character keeps its own texture colours, and a navy or black outfit
# against a black backdrop has no silhouette: its panel gets the lifted
# backdrop and a little more ambient fill than the white-on-black skeletons.
CHARACTER_AMBIENT = 0.34

# Image-only turntable: original still on the left, orbiting rig on the right.
# Elevation 10° looks down from about the character's own height; a slightly
# wider FOV and looser fill keep hands and feet inside the frame as the
# camera goes around.
ORBIT_ELEVATION_DEG = 10.0
ORBIT_FOV_DEG = 38.0
ORBIT_FILL = 0.86
ORBIT_SECONDS = 6.0
ORBIT_FPS = 30.0
ORBIT_PAD = 1.06


def video_size(video: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def image_size(image: Path) -> tuple[int, int]:
    import cv2

    frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot open image: {image}")
    h, w = frame.shape[:2]
    return w, h


def source_size(path: Path) -> tuple[int, int]:
    from .paths import is_image

    return image_size(path) if is_image(path) else video_size(path)


def panel_size(source: Path) -> tuple[int, int]:
    """Panel with the same aspect ratio as the input, even dimensions."""
    w, h = source_size(source)
    pw = int(round(PANEL_H * w / max(h, 1)))
    pw = min(max(pw, 2), MAX_PANEL_W)
    return pw + pw % 2, PANEL_H + PANEL_H % 2


class _OriginalReader:
    """Sequential reader that letterboxes each requested source frame."""

    def __init__(self, src: Path, width: int, height: int):
        import cv2

        self.blank = blank_panel(width, height)
        self.width, self.height = width, height
        self.cap = cv2.VideoCapture(str(src))
        self.next_idx = 0

    def frame(self, idx: int) -> np.ndarray:
        out = None
        while self.cap.isOpened() and self.next_idx <= idx:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.next_idx == idx:
                out = fit_letterbox(frame, self.width, self.height)
            self.next_idx += 1
        return self.blank.copy() if out is None else out

    def close(self) -> None:
        self.cap.release()


class _StreamWriter:
    """Raw BGR frames piped into ffmpeg, which encodes H.264 (via NVENC or libx264)
    and muxes the source video's audio over the rendered frame window.

    Falls back to a plain (silent) OpenCV writer when ffmpeg is unavailable.
    """

    def __init__(
        self,
        out_path: Path,
        fps: float,
        width: int,
        height: int,
        *,
        audio_src: Path | None = None,
        audio_start_s: float = 0.0,
        audio_dur_s: float | None = None,
        device: str = "cpu",
    ):
        from .cuda_accel import ffmpeg_video_encode_args

        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = None
        self.writer = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
            ]
            if audio_src is not None and Path(audio_src).exists():
                cmd += ["-ss", f"{audio_start_s:.6f}"]
                if audio_dur_s is not None:
                    cmd += ["-t", f"{audio_dur_s:.6f}"]
                # `0:v` is the rendered frames, `1:a:0?` the source audio if
                # it has any; -shortest stops the audio with the last frame.
                cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:a", "aac", "-b:a", "192k", "-shortest"]
            encode_args = ffmpeg_video_encode_args(device, crf=18)
            cmd += [*encode_args, "-movflags", "+faststart", str(self.out_path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            import cv2

            self.writer = cv2.VideoWriter(
                str(self.out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps), (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter for {self.out_path}")

    def write(self, frame: np.ndarray) -> None:
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self) -> Path:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed encoding {self.out_path}")
        if self.writer is not None:
            self.writer.release()
        return self.out_path


def _grid2x2(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray) -> np.ndarray:
    return np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])


def _write_png(path: Path, bgr: np.ndarray) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Cannot write {path}")
    return path


def _letterbox_still(image: Path, width: int, height: int) -> np.ndarray:
    import cv2

    frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot open image: {image}")
    return fit_letterbox(frame, width, height)


def _setup_view(
    anim: RetargetedAnimation,
    asset,
    calibration,
    width: int,
    height: int,
    *,
    device: str = "cpu",
):
    """Shared camera, floor and backdrops for the three rendered panels."""
    axes = calibration.rig_axes
    stride = max(1, anim.n // CAMERA_MESH_SAMPLES)
    probe = skin_frames(asset, anim, list(range(0, anim.n, stride)), device=device)
    frames_pts = [np.concatenate(v, axis=0) for v in probe]
    del probe
    soles = np.array([float(pts[:, 1].min()) for pts in frames_pts])
    mesh_pts = np.concatenate(frames_pts, axis=0)
    del frames_pts

    cam = build_perspective_camera(
        axes,
        np.concatenate(
            [all_points(anim.human_pos), all_points(anim.rig_skel_pos), mesh_pts],
            axis=0,
        ),
        width,
        height,
        fov_deg=CAMERA_FOV_DEG,
        azimuth_deg=CAMERA_AZIMUTH_DEG,
        elevation_deg=CAMERA_ELEVATION_DEG,
        fill=CAMERA_FILL,
    )
    del mesh_pts
    hips = np.stack([anim.rig_skel_pos[i]["rig:Hips"] for i in range(anim.n)])
    ground = Ground(
        y=float(np.percentile(soles, FLOOR_PERCENTILE)),
        cx=float(np.median(hips[:, 0])),
        cz=float(np.median(hips[:, 2])),
    )
    backdrop_dark = ground_backdrop(cam, ground, width, height, DARK_THEME)
    backdrop_grid = ground_backdrop(cam, ground, width, height, GRID_THEME)
    faces = [m.faces for m in asset.meshes]
    colors = [
        m.face_colors
        if m.face_colors is not None
        else MESH_COLORS_BGR.get(m.name, np.array([150.0, 150.0, 150.0]))
        for m in asset.meshes
    ]
    return cam, ground, backdrop_dark, backdrop_grid, faces, colors


def _draw_panels(
    anim: RetargetedAnimation,
    asset,
    calibration,
    i: int,
    *,
    original: np.ndarray,
    original_label: str,
    stamp: str,
    cam,
    ground,
    backdrop_dark,
    backdrop_grid,
    faces,
    colors,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    character = calibration.fbx_path.stem

    p_hum = backdrop_dark.copy()
    pos = anim.human_pos[i]
    draw_shadow(p_hum, cam, np.stack(list(pos.values())), ground, DARK_THEME)
    draw_skeleton(p_hum, pos, HUMAN_EDGES, cam, 2, DARK_THEME.joint)
    label_panel(p_hum, f"Human Skeleton{stamp}")

    p_mix = backdrop_dark.copy()
    pos = anim.rig_skel_pos[i]
    draw_shadow(p_mix, cam, np.stack(list(pos.values())), ground, DARK_THEME)
    draw_skeleton(p_mix, pos, RIG_EDGES, cam, 3, DARK_THEME.joint)
    label_panel(p_mix, f"Humanoid Skeleton{stamp}")

    p_char = backdrop_grid.copy()
    verts = skin_frames(asset, anim, [i], device=device)[0]
    draw_shadow(p_char, cam, np.concatenate(verts, axis=0), ground, GRID_THEME)
    draw_mesh(p_char, verts, faces, colors, cam, CHARACTER_AMBIENT, device=device)
    label_panel(p_char, f"{character}{stamp}")

    p_orig = original.copy()
    label_panel(p_orig, f"{original_label}{stamp}")
    return p_orig, p_mix, p_hum, p_char


def render_run_images(
    anim: RetargetedAnimation,
    asset,
    calibration,
    source_image: Path,
    out_dir: Path,
    *,
    original_label: str = "Original Photo",
    verbose: bool = True,
    device: str = "cpu",
) -> dict[str, Path]:
    """Write the four stills into `out_dir` (images/ of a still run).
 
    Uses the middle hold frame: the centre of the repeated hold sequence is
    the most settled pose.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = panel_size(source_image)
    cam, ground, backdrop_dark, backdrop_grid, faces, colors = _setup_view(
        anim, asset, calibration, width, height, device=device
    )
    if verbose:
        print(f"  camera eye {np.round(cam.eye, 3)}  fov {cam.fov_deg:.1f} deg"
              f"  floor y {ground.y:+.4f} m")

    i = anim.n // 2
    p_orig, p_mix, p_hum, p_char = _draw_panels(
        anim, asset, calibration, i,
        original=_letterbox_still(source_image, width, height),
        original_label=original_label,
        stamp="",
        cam=cam, ground=ground,
        backdrop_dark=backdrop_dark, backdrop_grid=backdrop_grid,
        faces=faces, colors=colors,
        device=device,
    )
    out = {
        "human_skeleton": _write_png(out_dir / "human_skeleton.png", p_hum),
        "humanoid_skeleton": _write_png(out_dir / "humanoid_skeleton.png", p_mix),
        "humanoid_character": _write_png(out_dir / "humanoid_character.png", p_char),
        "compare": _write_png(out_dir / "compare.png", _grid2x2(p_orig, p_mix, p_hum, p_char)),
    }
    if verbose:
        print("rendering 360 compare")
    out["before_after_360_compare"] = render_orbit_compare(
        anim, asset, calibration, source_image, out_dir, verbose=verbose, device=device
    )
    if verbose:
        for path in out.values():
            print(f"saved {path}")
    return out


def _orbit_camera(
    axes: dict,
    target: np.ndarray,
    dist: float,
    azimuth_deg: float,
    width: int,
    height: int,
    *,
    elevation_deg: float = ORBIT_ELEVATION_DEG,
    fov_deg: float = ORBIT_FOV_DEG,
) -> PerspectiveCamera:
    """Pinhole eye on a circle around `target`, looking down at `elevation_deg`."""
    up = normalize(np.asarray(axes["up"], dtype=np.float64))
    fwd = normalize(np.asarray(axes["forward"], dtype=np.float64))
    right = normalize(np.asarray(axes["right"], dtype=np.float64))
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    offset = normalize(np.cos(el) * (np.cos(az) * fwd + np.sin(az) * right) + np.sin(el) * up)
    look = -offset
    cam_right = normalize(np.cross(look, up))
    cam_up = normalize(np.cross(cam_right, look))
    focal = (height * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    return PerspectiveCamera(
        target + offset * dist, look, cam_right, cam_up, float(focal), width, height
    )


def _fit_orbit_distance(
    axes: dict,
    points: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
) -> float:
    """One radius that keeps the whole mesh on screen at every heading.

    Starts at the distance that puts the eye at about the character's height
    (elevation 10° looking at the torso centre), then grows if any heading
    would clip a hand or a foot.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    up = normalize(np.asarray(axes["up"], dtype=np.float64))
    along = pts @ up
    head = float(along.max())
    mid = float(target @ up)
    el = np.radians(ORBIT_ELEVATION_DEG)
    dist = max((head - mid) / max(np.sin(el), 1e-3), 1e-3)
    center = pts.mean(axis=0)
    padded = center + (pts - center) * ORBIT_PAD
    for az in np.linspace(0.0, 360.0, 12, endpoint=False):
        for _ in range(60):
            cam = _orbit_camera(axes, target, dist, float(az), width, height)
            uv = cam.project(padded)
            lo, hi = uv.min(axis=0), uv.max(axis=0)
            need = max(
                (hi[0] - lo[0]) / (ORBIT_FILL * width),
                (hi[1] - lo[1]) / (ORBIT_FILL * height),
                1.0,
            )
            inside = (lo >= 0.0).all() and (hi < (width, height)).all()
            if need <= 1.0 and inside:
                break
            dist *= max(need, 1.02)
    return float(dist)


def render_orbit_compare(
    anim: RetargetedAnimation,
    asset,
    calibration,
    source_image: Path,
    out_dir: Path,
    *,
    verbose: bool = True,
    device: str = "cpu",
) -> Path:
    """Left = original photo, right = 10° orbit of the skinned rig. Image runs only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = panel_size(source_image)
    i = anim.n // 2
    verts = skin_frames(asset, anim, [i], device=device)[0]
    mesh_pts = np.concatenate(verts, axis=0)
    axes = calibration.rig_axes
    target = 0.5 * (mesh_pts.min(axis=0) + mesh_pts.max(axis=0))
    dist = _fit_orbit_distance(axes, mesh_pts, target, width, height)
    hips = anim.rig_skel_pos[i]["rig:Hips"]
    ground = Ground(
        y=float(np.percentile(mesh_pts[:, 1], FLOOR_PERCENTILE)),
        cx=float(hips[0]),
        cz=float(hips[2]),
    )

    faces = [m.faces for m in asset.meshes]
    colors = [
        m.face_colors
        if m.face_colors is not None
        else MESH_COLORS_BGR.get(m.name, np.array([150.0, 150.0, 150.0]))
        for m in asset.meshes
    ]
    left = _letterbox_still(source_image, width, height)
    label_panel(left, "Original Photo")
    character = calibration.fbx_path.stem
    n = max(2, int(round(ORBIT_SECONDS * ORBIT_FPS)))
    out_path = out_dir / "before_after_360_compare.mp4"
    writer = _StreamWriter(out_path, ORBIT_FPS, 2 * width, height, device=device)
    if verbose:
        print(f"  orbit dist {dist:.3f} m  {n} frames @ {ORBIT_FPS:g} fps"
              f"  elev {ORBIT_ELEVATION_DEG:g} deg")
    try:
        for k in range(n):
            az = 360.0 * k / n
            cam = _orbit_camera(axes, target, dist, az, width, height)
            right = ground_backdrop(cam, ground, width, height, GRID_THEME)
            draw_shadow(right, cam, mesh_pts, ground, GRID_THEME)
            draw_mesh(right, verts, faces, colors, cam, CHARACTER_AMBIENT, device=device)
            label_panel(right, character)
            writer.write(np.hstack([left, right]))
            if verbose and (k == 0 or (k + 1) % 30 == 0 or k == n - 1):
                print(f"  orbit {k + 1}/{n}")
    finally:
        writer.close()
    return out_path


def render_run_videos(
    anim: RetargetedAnimation,
    asset,
    calibration,
    source_video: Path,
    out_dir: Path,
    *,
    original_label: str = "Original Video",
    verbose: bool = True,
    device: str = "cpu",
) -> dict[str, Path]:
    """Write the four result videos into `out_dir` (videos/ of a run)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = panel_size(source_video)
    fps = anim.fps
    cam, ground, backdrop_dark, backdrop_grid, faces, colors = _setup_view(
        anim, asset, calibration, width, height, device=device
    )
    if verbose:
        print(f"  camera eye {np.round(cam.eye, 3)}  fov {cam.fov_deg:.1f} deg"
              f"  floor y {ground.y:+.4f} m")

    audio_start = anim.frames[0] / fps
    audio_dur = anim.n / fps

    def writer(name: str, w: int, h: int) -> _StreamWriter:
        return _StreamWriter(
            out_dir / name, fps, w, h,
            audio_src=source_video,
            audio_start_s=audio_start,
            audio_dur_s=audio_dur,
            device=device,
        )

    writers = {
        "human_skeleton": writer("human_skeleton.mp4", width, height),
        "humanoid_skeleton": writer("humanoid_skeleton.mp4", width, height),
        "humanoid_character": writer("humanoid_character.mp4", width, height),
        "compare": writer("compare.mp4", 2 * width, 2 * height),
    }
    originals = _OriginalReader(source_video, width, height)
    try:
        for i in range(anim.n):
            t = anim.frames[i] / fps
            p_orig, p_mix, p_hum, p_char = _draw_panels(
                anim, asset, calibration, i,
                original=originals.frame(anim.frames[i]),
                original_label=original_label,
                stamp=f"  f{anim.frames[i]}  t={t:.2f}s",
                cam=cam, ground=ground,
                backdrop_dark=backdrop_dark, backdrop_grid=backdrop_grid,
                faces=faces, colors=colors,
                device=device,
            )
            writers["human_skeleton"].write(p_hum)
            writers["humanoid_skeleton"].write(p_mix)
            writers["humanoid_character"].write(p_char)
            writers["compare"].write(_grid2x2(p_orig, p_mix, p_hum, p_char))

            if verbose and (i == 0 or (i + 1) % 100 == 0 or i == anim.n - 1):
                print(f"  render {i + 1}/{anim.n}")
    finally:
        originals.close()

    out = {name: w.close() for name, w in writers.items()}
    if verbose:
        for path in out.values():
            print(f"saved {path}")
    return out


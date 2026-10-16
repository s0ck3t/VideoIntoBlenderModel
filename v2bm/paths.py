"""Single source of truth for project paths: assets, weights, outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS = PROJECT_ROOT / "assets"
RIGS_DIR = ASSETS / "rigs"
DEFAULT_RIG = RIGS_DIR / "humanoid_rig.fbx"
VIDEO_DIR = ASSETS / "video"
IMAGE_DIR = ASSETS / "image"

WEIGHTS = PROJECT_ROOT / "weights"
OUTPUTS = PROJECT_ROOT / "outputs"

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _list_media(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [
        p
        for p in folder.iterdir()
        if p.suffix.lower() in suffixes and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def is_image(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def list_videos() -> list[Path]:
    """User-supplied videos in assets/video/, most recently added first."""
    return _list_media(VIDEO_DIR, VIDEO_SUFFIXES)


def latest_video() -> Path | None:
    """The video most recently placed into assets/video/ (by file mtime)."""
    videos = list_videos()
    return videos[0] if videos else None


def list_images() -> list[Path]:
    """User-supplied stills in assets/image/, most recently added first."""
    return _list_media(IMAGE_DIR, IMAGE_SUFFIXES)


def latest_image() -> Path | None:
    """The still most recently placed into assets/image/ (by file mtime)."""
    images = list_images()
    return images[0] if images else None


def new_run_dir(source: Path, now: datetime | None = None) -> Path:
    """Create outputs/<YYYYMMDD_HHMMSS>_<source-stem>/ for one `v2bm run`."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS / f"{stamp}_{source.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


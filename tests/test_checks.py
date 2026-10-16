"""Unit and regression tests for v2bm.checks.

Verifies input validation, CC0 character rig verification, Google MediaPipe
Pose person detection, and error diagnostics under strictly permissive licensing.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from v2bm import checks, paths

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_VIDEO = REPO_ROOT / "docs" / "assets" / "demo.mp4"
DEMO_IMAGE = REPO_ROOT / "assets" / "image" / "urban_dancer_pose.png"
HUMANOID_RIG_FBX = REPO_ROOT / "assets" / "rigs" / "humanoid_rig.fbx"


def test_ensure_mediapipe_model() -> None:
    """Verify that ensure_mediapipe_model resolves to a valid existing task file."""
    model_path = checks.ensure_mediapipe_model()
    assert model_path.exists(), f"MediaPipe model file missing at {model_path}"
    assert model_path.stat().st_size > 1_000_000, "Model file size unexpectedly small"


def test_check_video_valid() -> None:
    """Verify check_video accepts single-person video without error."""
    if not DEMO_VIDEO.exists():
        pytest.skip(f"Reference video not found at {DEMO_VIDEO}")
    result = checks.check_video(DEMO_VIDEO)
    assert result is None, f"Expected valid video, got error: {result}"


def test_check_video_nonexistent(tmp_path: Path) -> None:
    """Verify check_video reports clear error on nonexistent video."""
    bad_video = tmp_path / "missing.mp4"
    result = checks.check_video(bad_video)
    assert result is not None
    assert "cannot be decoded as video" in result


def test_check_image_valid() -> None:
    """Verify check_image accepts single-person still image without error."""
    if not DEMO_IMAGE.exists():
        pytest.skip(f"Reference image not found at {DEMO_IMAGE}")
    result = checks.check_image(DEMO_IMAGE)
    assert result is None, f"Expected valid image, got error: {result}"


def test_check_image_nonexistent(tmp_path: Path) -> None:
    """Verify check_image reports clear error on nonexistent image."""
    bad_image = tmp_path / "missing.png"
    result = checks.check_image(bad_image)
    assert result is not None
    assert "cannot be decoded as an image" in result


def test_check_image_blank(tmp_path: Path) -> None:
    """Verify check_image rejects empty image with no detected persons."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_path = tmp_path / "blank.png"
    cv2.imwrite(str(blank_path), blank_frame)

    result = checks.check_image(blank_path)
    assert result is not None
    assert "no person was detected" in result


def test_check_cc0_rig_valid() -> None:
    """Verify check_cc0_rig validates the generated humanoid_rig.fbx rig."""
    if not HUMANOID_RIG_FBX.exists():
        pytest.skip(f"CC0 rig not found at {HUMANOID_RIG_FBX}")
    result = checks.check_cc0_rig(HUMANOID_RIG_FBX)
    assert result is None, f"Expected valid rig, got error: {result}"


def test_check_cc0_rig_missing(tmp_path: Path) -> None:
    """Verify check_cc0_rig returns helpful hint when rig is missing."""
    missing_rig = tmp_path / "missing.fbx"
    result = checks.check_cc0_rig(missing_rig)
    assert result is not None
    assert "is missing" in result
    assert "scripts/generate_humanoid_rig.py" in result


def test_check_rig_not_binary(tmp_path: Path) -> None:
    """Verify check_rig detects ASCII FBX and warns user."""
    ascii_fbx = tmp_path / "ascii.fbx"
    ascii_fbx.write_text("; FBX 7.4.0 project file\n", encoding="utf-8")

    result = checks.check_rig(ascii_fbx)
    assert result is not None
    assert "not a *binary* FBX" in result


def test_check_skeleton_npz_valid(tmp_path: Path) -> None:
    """Verify check_skeleton_npz accepts compliant skeleton motion file."""
    npz_path = tmp_path / "skeleton_motion.npz"
    np.savez(
        npz_path,
        joints_3d=np.zeros((10, 24, 3), dtype=np.float32),
        joint_names=np.array(["pelvis"] * 24),
        fps=np.float32(30.0),
    )
    result = checks.check_skeleton_npz(npz_path)
    assert result is None, f"Expected valid skeleton npz, got: {result}"


def test_check_skeleton_npz_missing_keys(tmp_path: Path) -> None:
    """Verify check_skeleton_npz detects missing arrays."""
    npz_path = tmp_path / "bad_skeleton.npz"
    np.savez(npz_path, foo=np.zeros(5))

    result = checks.check_skeleton_npz(npz_path)
    assert result is not None
    assert "is missing" in result

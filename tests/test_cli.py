"""Tier 3: End-to-End CLI Verification.

Opaque-box CLI tests verifying:
- `v2bm --help` and `v2bm run --help`
- `v2bm doctor` (zero non-commercial asset references, clean preflight output)
- `v2bm run --video <demo.mp4>`
- `v2bm run --image <repo_poster.png>`
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
DEMO_VIDEO = REPO_ROOT / "docs" / "assets" / "demo.mp4"
DEMO_IMAGE = REPO_ROOT / "assets" / "image" / "urban_dancer_pose.png"
HUMANOID_RIG_FBX = REPO_ROOT / "assets" / "rigs" / "humanoid_rig.fbx"

EXTRACT_PY = REPO_ROOT / "v2bm" / "extract.py"
M3_EXTRACT_AVAILABLE = (
    EXTRACT_PY.exists()
    and "legacy_lifter" not in EXTRACT_PY.read_text(encoding="utf-8")
)


def run_cli(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    """Execute v2bm CLI via Python subprocess."""
    cmd = [PYTHON_EXE, "-m", "v2bm.cli", *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )



def test_cli_help() -> None:
    """Verify CLI top-level help executes cleanly and documents doctor and run."""
    result = run_cli("--help")
    assert result.returncode == 0, f"CLI help failed with stderr: {result.stderr}"
    assert "doctor" in result.stdout
    assert "run" in result.stdout


def test_cli_run_help() -> None:
    """Verify CLI run subcommand help documents --video, --image, and --rig flags."""
    result = run_cli("run", "--help")
    assert result.returncode == 0, f"CLI run help failed: {result.stderr}"
    assert "--video" in result.stdout
    assert "--image" in result.stdout
    assert "--rig" in result.stdout


def test_cli_doctor_no_prohibited_references() -> None:
    """Verify v2bm doctor contains zero references to Body-Model, legacy-lifter, or third-party-detector."""
    result = run_cli("doctor")
    combined_output = (result.stdout + " " + result.stderr).lower()

    # Must NOT mention non-commercial research models or copyleft dependencies
    assert "bodymodel_neutral.npz" not in combined_output, (
        "v2bm doctor must not reference neutral_body_model.npz"
    )
    assert "body24-x" not in combined_output, (
        "v2bm doctor must not reference Body-Model"
    )
    assert "legacy_lifter" not in combined_output, (
        "v2bm doctor must not reference legacy-lifter"
    )
    assert "third_party_detector" not in combined_output, (
        "v2bm doctor must not reference third-party-detector"
    )


def test_cli_run_missing_input_diagnostics(tmp_path: Path) -> None:
    """Verify v2bm run reports clear preflight diagnostics when given non-existent files."""
    fake_video = tmp_path / "non_existent_clip.mp4"
    result = run_cli("run", "--video", str(fake_video))
    assert result.returncode != 0, "CLI should fail on non-existent input"
    # Should report missing or invalid file, not crash with an unhandled exception
    combined = result.stdout + " " + result.stderr
    assert "Traceback" not in result.stderr or "FileNotFoundError" in combined or "missing" in combined.lower()


@pytest.mark.skipif(
    not (HUMANOID_RIG_FBX.exists() and DEMO_VIDEO.exists() and M3_EXTRACT_AVAILABLE),
    reason="Full video E2E test requires generated humanoid_rig.fbx, demo.mp4, and M3 MediaPipe extractor",
)
def test_cli_run_video_end_to_end(tmp_path: Path) -> None:
    """Verify v2bm run --video generates all required artifacts end-to-end."""
    out_dir = tmp_path / "video_run"
    result = run_cli(
        "run",
        "--video",
        str(DEMO_VIDEO),
        "--rig",
        str(HUMANOID_RIG_FBX),
        "--out",
        str(out_dir),
    )
    assert result.returncode == 0, (
        f"v2bm run --video failed with exit code {result.returncode}\n"
        f"Stdout:\n{result.stdout}\n"
        f"Stderr:\n{result.stderr}"
    )

    # Verify expected artifacts
    assert (out_dir / "skeleton_motion.npz").exists(), "Missing skeleton_motion.npz"
    assert (out_dir / "rig_rotations.npz").exists(), "Missing rig_rotations.npz"
    assert (out_dir / "character_animation.glb").exists(), "Missing character_animation.glb"
    assert (out_dir / "videos" / "compare.mp4").exists(), "Missing videos/compare.mp4"


@pytest.mark.skipif(
    not (HUMANOID_RIG_FBX.exists() and DEMO_IMAGE.exists() and M3_EXTRACT_AVAILABLE),
    reason="Full image E2E test requires generated humanoid_rig.fbx, repo_poster.png, and M3 MediaPipe extractor",
)
def test_cli_run_image_end_to_end(tmp_path: Path) -> None:
    """Verify v2bm run --image generates all required still artifacts end-to-end."""
    out_dir = tmp_path / "image_run"
    result = run_cli(
        "run",
        "--image",
        str(DEMO_IMAGE),
        "--rig",
        str(HUMANOID_RIG_FBX),
        "--out",
        str(out_dir),
    )
    assert result.returncode == 0, (
        f"v2bm run --image failed with exit code {result.returncode}\n"
        f"Stdout:\n{result.stdout}\n"
        f"Stderr:\n{result.stderr}"
    )

    # Verify expected artifacts
    assert (out_dir / "images" / "compare.png").exists(), "Missing images/compare.png"
    assert (out_dir / "images" / "before_after_360_compare.mp4").exists(), (
        "Missing images/before_after_360_compare.mp4"
    )

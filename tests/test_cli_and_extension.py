"""Milestone 4 Verification: CLI, Preflight Checks & Blender Extension Integration.

Comprehensive tests verifying:
- `v2bm run` argument parsing (`--capture-face`, `--no-face`, `--framing`)
- `v2bm doctor` reporting of pose and face model weights, licences, sizes, and acceleration
- `v2bm.checks` framing behaviour for full-body vs portrait close-ups
- Blender Extension manifest, properties, i18n translations, and command line construction
- Blender Extension packaging (`dist/v2bm-0.1.0.zip`) and official Blender CLI validation
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

# Setup mock for bpy prior to importing any blender extension module
class MockPropertyGroup:
    pass

class MockScene:
    pass

mock_bpy = MagicMock()
mock_bpy_props = MagicMock()
mock_bpy.props = mock_bpy_props
mock_bpy_types = MagicMock()
mock_bpy_types.PropertyGroup = MockPropertyGroup
mock_bpy_types.Scene = MockScene

sys.modules["bpy"] = mock_bpy
sys.modules["bpy.props"] = mock_bpy_props
sys.modules["bpy.types"] = mock_bpy_types

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
URBAN_DANCER_IMAGE = REPO_ROOT / "assets" / "image" / "urban_dancer_pose.png"
KUNG_FU_VIDEO = REPO_ROOT / "assets" / "video" / "kung_fu_haikou.mp4"
BLENDER_MANIFEST = REPO_ROOT / "blender_extension" / "v2bm_blender" / "blender_manifest.toml"


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


# ==============================================================================
# 1. CLI Argument Parsing Tests
# ==============================================================================


def test_cli_argument_parsing() -> None:
    """Verify --capture-face, --no-face, and --framing arguments and defaults."""
    from v2bm.cli import build_parser

    parser = build_parser()

    # Test default values
    args_default = parser.parse_args(["run"])
    assert args_default.capture_face is True, "Default capture_face should be True"
    assert args_default.framing == "auto", "Default framing should be 'auto'"

    # Test explicit --capture-face
    args_face = parser.parse_args(["run", "--capture-face"])
    assert args_face.capture_face is True, "--capture-face should set capture_face to True"

    # Test --no-face
    args_no_face = parser.parse_args(["run", "--no-face"])
    assert args_no_face.capture_face is False, "--no-face should set capture_face to False"

    # Test framing options
    for framing_choice in ("auto", "full_body", "portrait"):
        args_framing = parser.parse_args(["run", "--framing", framing_choice])
        assert args_framing.framing == framing_choice

    # Test invalid framing choice
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--framing", "invalid_framing_mode"])

    # Test help text documents the new flags
    result_help = run_cli("run", "--help")
    assert result_help.returncode == 0
    assert "--capture-face" in result_help.stdout
    assert "--no-face" in result_help.stdout
    assert "--framing" in result_help.stdout
    assert "portrait" in result_help.stdout


# ==============================================================================
# 2. CLI Doctor Face Reporting Tests
# ==============================================================================


def test_cli_doctor_face_reporting() -> None:
    """Verify v2bm doctor reports pose and face models, Apache-2.0 licences, sizes, and acceleration."""
    result = run_cli("doctor")
    assert result.returncode == 0, f"v2bm doctor failed with code {result.returncode}:\n{result.stderr}"

    output = result.stdout
    combined = (output + "\n" + result.stderr).lower()

    # Verify pose and face model reporting
    assert "pose_landmarker_full.task" in output, "doctor must report pose_landmarker_full.task"
    assert "face_landmarker.task" in output, "doctor must report face_landmarker.task"

    # Verify Apache-2.0 licence reporting
    assert "apache-2.0" in combined, "doctor must document Apache-2.0 licence"

    # Verify file sizes reported in MB
    assert "mb" in combined, "doctor must report model file sizes in MB"

    # Verify hardware acceleration reporting
    assert "face acceleration" in output, "doctor must report facial hardware acceleration status"
    assert (
        "cupy gpu acceleration" in combined or "cpu numpy fallback" in combined
    ), "doctor must report CuPy GPU acceleration or CPU NumPy fallback"

    # Verify total model weights note (~12.5 - 12.8 MB)
    assert "inference weights:" in output, "doctor must output inference weights summary"
    assert "mediapipe" in combined, "weights summary must mention MediaPipe"

    # Verify absence of non-commercial or prohibited references
    assert "bodymodel" not in combined
    assert "legacy_lifter" not in combined
    assert "smpl" not in combined
    assert "flame" not in combined
    assert "deca" not in combined


# ==============================================================================
# 3. Framing Checks Behaviour Tests
# ==============================================================================


def test_framing_checks_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify check_image and check_video framing handling and top-level doctor alias."""
    import v2bm.checks as checks

    # Verify top-level doctor alias exists and is callable
    assert hasattr(checks, "doctor"), "v2bm.checks must provide doctor() alias"
    assert callable(checks.doctor), "v2bm.checks.doctor must be callable"

    # Verify check_image on full-body image
    assert URBAN_DANCER_IMAGE.exists(), f"Missing test image: {URBAN_DANCER_IMAGE}"
    err_auto = checks.check_image(URBAN_DANCER_IMAGE, framing_mode="auto")
    assert err_auto is None, f"check_image(auto) failed: {err_auto}"

    err_full = checks.check_image(URBAN_DANCER_IMAGE, framing_mode="full_body")
    assert err_full is None, f"check_image(full_body) failed: {err_full}"

    # On full-body image with no direct full-frame face detected, portrait reports hint
    err_portrait = checks.check_image(URBAN_DANCER_IMAGE, framing_mode="portrait")
    assert err_portrait is not None
    assert "face" in err_portrait.lower()

    # Create synthetic test frame for cropped shot without lower limbs
    test_img = np.zeros((480, 480, 3), dtype=np.uint8)
    test_img_path = tmp_path / "portrait_test.png"
    cv2.imwrite(str(test_img_path), test_img)

    # When no full body person is detected (lower limbs absent)
    monkeypatch.setattr(checks, "_count_people", lambda *args, **kwargs: 0)

    # If face is detected (1 face)
    monkeypatch.setattr(checks, "_count_faces", lambda *args, **kwargs: 1)

    # full_body must reject because lower limbs/body are missing
    err_cropped_full = checks.check_image(test_img_path, framing_mode="full_body")
    assert err_cropped_full is not None
    assert "person" in err_cropped_full.lower()

    # portrait mode must succeed without rejecting due to absent lower limbs
    err_cropped_portrait = checks.check_image(test_img_path, framing_mode="portrait")
    assert err_cropped_portrait is None, f"portrait mode should accept single face: {err_cropped_portrait}"

    # auto mode must fall back cleanly to portrait and pass
    err_cropped_auto = checks.check_image(test_img_path, framing_mode="auto")
    assert err_cropped_auto is None, f"auto mode should accept portrait fallback: {err_cropped_auto}"

    # If multiple faces detected (2 faces), portrait mode must reject crowded shot
    monkeypatch.setattr(checks, "_count_faces", lambda *args, **kwargs: 2)
    err_crowded = checks.check_image(test_img_path, framing_mode="portrait")
    assert err_crowded is not None
    assert "2 faces" in err_crowded

    # Verify check_video on existing video
    if KUNG_FU_VIDEO.exists():
        monkeypatch.undo()
        err_vid_auto = checks.check_video(KUNG_FU_VIDEO, framing_mode="auto")
        assert err_vid_auto is None, f"check_video(auto) failed: {err_vid_auto}"

        err_vid_full = checks.check_video(KUNG_FU_VIDEO, framing_mode="full_body")
        assert err_vid_full is None, f"check_video(full_body) failed: {err_vid_full}"


# ==============================================================================
# 4. Blender Extension Properties & Manifest Tests
# ==============================================================================


def test_blender_extension_properties_and_manifest() -> None:
    """Verify blender_manifest.toml, build_run_argv, i18n keys, and V2BMProperties."""
    # 4.1 Verify blender_manifest.toml
    assert BLENDER_MANIFEST.exists(), f"Missing manifest at {BLENDER_MANIFEST}"
    with open(BLENDER_MANIFEST, "rb") as mf:
        manifest = tomllib.load(mf)

    assert manifest.get("id") == "v2bm"
    assert manifest.get("schema_version") == "1.0.0"
    assert manifest.get("version") == "0.1.0"
    assert manifest.get("type") == "add-on"
    assert "license" in manifest

    # 4.2 Verify build_run_argv
    from blender_extension.v2bm_blender.services.backend import build_run_argv

    cmd_default = build_run_argv(
        PYTHON_EXE,
        source_type="video",
        source=KUNG_FU_VIDEO,
        rig="assets/rigs/humanoid_rig.fbx",
        device="cpu",
        output_dir="outputs/test_job",
        progress_jsonl="outputs/test_job/progress.jsonl",
        preview=True,
        capture_face=True,
        framing="auto",
    )
    assert "--capture-face" in cmd_default
    assert "--no-face" not in cmd_default
    assert "--framing" in cmd_default
    assert cmd_default[cmd_default.index("--framing") + 1] == "auto"
    assert "--no-preview" not in cmd_default

    # Test capture_face=False
    cmd_no_face = build_run_argv(
        PYTHON_EXE,
        source_type="image",
        source=URBAN_DANCER_IMAGE,
        rig="assets/rigs/humanoid_rig.fbx",
        device="cuda",
        output_dir="outputs/test_job",
        progress_jsonl="outputs/test_job/progress.jsonl",
        preview=False,
        capture_face=False,
        framing="portrait",
    )
    assert "--no-face" in cmd_no_face
    assert "--capture-face" not in cmd_no_face
    assert "--framing" in cmd_no_face
    assert cmd_no_face[cmd_no_face.index("--framing") + 1] == "portrait"
    assert "--no-preview" in cmd_no_face
    assert "--image" in cmd_no_face

    # Test invalid parameters raise ValueError
    with pytest.raises(ValueError):
        build_run_argv(PYTHON_EXE, source_type="audio", source="a", rig="r", device="cpu", output_dir="o", progress_jsonl="p", preview=False)
    with pytest.raises(ValueError):
        build_run_argv(PYTHON_EXE, source_type="video", source="v", rig="r", device="tpu", output_dir="o", progress_jsonl="p", preview=False)

    # 4.3 Verify i18n keys
    from blender_extension.v2bm_blender.i18n import STRINGS

    required_keys = (
        "capture_face",
        "framing",
        "framing_auto",
        "framing_full_body",
        "framing_portrait",
    )
    for lang in ("en", "zh-CN", "ko", "ja", "de", "ru", "ar"):
        assert lang in STRINGS, f"Language {lang} missing from STRINGS"
        for key in required_keys:
            assert key in STRINGS[lang], f"Key {key!r} missing for lang {lang}"
            assert isinstance(STRINGS[lang][key], str) and STRINGS[lang][key].strip()

    # 4.4 Verify V2BMProperties registration attributes
    from blender_extension.v2bm_blender.properties import V2BMProperties

    assert hasattr(V2BMProperties, "__annotations__")
    annotations = V2BMProperties.__annotations__
    assert "capture_face" in annotations, "V2BMProperties must include capture_face"
    assert "framing" in annotations, "V2BMProperties must include framing"


# ==============================================================================
# 5. Extension Packaging & Validation Tests
# ==============================================================================


def test_extension_packaging_and_validation(tmp_path: Path) -> None:
    """Verify package_blender_extension creates valid zip and passes Blender validation."""
    from scripts.package_blender_extension import (
        build_zip,
        check_zip_structure,
        maybe_blender_validate,
    )

    zip_dest = tmp_path / "v2bm-test.zip"
    created_zip = build_zip(dest=zip_dest)
    assert created_zip.exists(), f"Failed to build zip at {created_zip}"
    assert created_zip.stat().st_size > 5000, "Zip archive suspiciously small"

    # Verify zip internal structure
    errors = check_zip_structure(created_zip)
    assert not errors, f"Zip structure errors: {errors}"

    with zipfile.ZipFile(created_zip, "r") as archive:
        names = set(archive.namelist())
        assert "blender_manifest.toml" in names, "blender_manifest.toml missing from zip root"
        assert "__init__.py" in names, "__init__.py missing from zip root"
        assert "properties.py" in names, "properties.py missing from zip root"
        assert "i18n.py" in names, "i18n.py missing from zip root"
        assert not any(n.endswith(".DS_Store") for n in names)
        assert not any(n.endswith(".gitkeep") for n in names)

    # Run Blender CLI validation
    val_result = maybe_blender_validate(created_zip)
    if val_result is not None:
        assert "ok" in val_result.lower(), f"Blender extension validation reported error: {val_result}"


# ==============================================================================
# 6. Preflight Missing Face Weights Robustness Tests
# ==============================================================================


def test_preflight_missing_face_weights_with_no_face(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify preflight succeeds cleanly when --no-face is passed even if face weights are missing."""
    from unittest.mock import patch
    from v2bm.cli import build_parser, preflight

    parser = build_parser()
    video = KUNG_FU_VIDEO if KUNG_FU_VIDEO.exists() else Path("assets/video/kung_fu_haikou.mp4")
    rig = REPO_ROOT / "assets" / "rigs" / "humanoid_rig.fbx"

    nonexistent_model = tmp_path / "nonexistent_face_landmarker.task"
    with patch("v2bm.checks.ensure_mediapipe_face_model", return_value=nonexistent_model):
        args = parser.parse_args(["run", "--video", str(video), "--rig", str(rig), "--no-face"])
        v, r, s, img = preflight(args)
        assert v is not None and v.exists(), "preflight should return valid video path"
        assert r is not None and r.exists(), "preflight should return valid rig path"
        assert s is None, "skeleton should be None"
        assert img is None, "image should be None"


def test_preflight_missing_face_weights_with_capture_face(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify preflight cleanly reports missing face model and exits with code 1 without unhandled exception."""
    from unittest.mock import patch
    from v2bm.cli import build_parser, preflight

    parser = build_parser()
    video = KUNG_FU_VIDEO if KUNG_FU_VIDEO.exists() else Path("assets/video/kung_fu_haikou.mp4")
    rig = REPO_ROOT / "assets" / "rigs" / "humanoid_rig.fbx"

    nonexistent_model = tmp_path / "nonexistent_face_landmarker.task"
    with patch("v2bm.checks.ensure_mediapipe_face_model", return_value=nonexistent_model):
        args = parser.parse_args(["run", "--video", str(video), "--rig", str(rig), "--capture-face"])
        with pytest.raises(SystemExit) as exc_info:
            preflight(args)
        assert exc_info.value.code == 1, f"Expected exit code 1, got {exc_info.value.code}"

    captured = capsys.readouterr()
    combined = captured.out + "\n" + captured.err
    assert "MediaPipe face" in combined, "Output must mention MediaPipe face"
    assert "face_landmarker.task" in combined, "Output must mention face_landmarker.task"
    assert "Traceback" not in combined, "Must not dump Python traceback"


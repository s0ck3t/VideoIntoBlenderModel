"""Tier 4: Headless Blender 5.2.2 LTS Automated Verification Test.

Invokes the headless Blender verification script via `blender-launcher.exe`,
inspects the disk log and structured JSON result, and asserts STATUS: PASS.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BLENDER_LAUNCHER = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Microsoft"
    / "WindowsApps"
    / "blender-launcher.exe"
)
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_blender_glb.py"

# Candidates for GLB verification
CANDIDATE_GLBS = [
    REPO_ROOT / "outputs" / "character_animation.glb",
    REPO_ROOT / "outputs" / "test_verify.glb",
    REPO_ROOT / "character_animation.glb",
]


def find_test_glb() -> Path | None:
    for p in CANDIDATE_GLBS:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


@pytest.mark.skipif(
    not BLENDER_LAUNCHER.exists(),
    reason="Blender 5.2.2 LTS launcher missing at expected Windows path",
)
def test_headless_blender_glb_verification(tmp_path: Path) -> None:
    """Execute verify_blender_glb.py inside headless Blender 5.2.2 LTS and verify PASS."""
    target_glb = find_test_glb()
    if target_glb is None:
        pytest.skip("No candidate .glb file found for Blender LTS verification")

    log_path = tmp_path / "blender_test.log"
    json_path = tmp_path / "blender_test.json"
    render_path = tmp_path / "blender_render.png"

    cmd = [
        "powershell",
        "-Command",
        (
            f"Start-Process -FilePath '{BLENDER_LAUNCHER}' "
            f"-ArgumentList '--background', '--python', '{VERIFY_SCRIPT}', '--', "
            f"'--glb', '{target_glb}', "
            f"'--log', '{log_path}', "
            f"'--json', '{json_path}', "
            f"'--render', '{render_path}' -Wait"
        ),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"PowerShell execution failed: {result.stderr}"

    # Verify log output
    assert log_path.exists(), f"Blender log file missing at {log_path}"
    log_content = log_path.read_text(encoding="utf-8")
    assert "STATUS: PASS" in log_content, (
        f"Blender verification failed. Log content:\n{log_content}"
    )

    # Verify JSON output
    assert json_path.exists(), f"Blender JSON output missing at {json_path}"
    with open(json_path, "r", encoding="utf-8") as jf:
        data = json.load(jf)

    assert data.get("status") == "PASS", f"Expected status PASS, got: {data}"
    assert data["armature"]["bone_count"] in (25, 61), (
        f"Expected 25 or 61 bones, got {data['armature']['bone_count']}"
    )
    assert data["animation"]["fcurves_count"] > 0, "Expected non-empty animation curves"
    assert render_path.exists() and render_path.stat().st_size > 0, (
        "Rendered verification frame missing or empty"
    )

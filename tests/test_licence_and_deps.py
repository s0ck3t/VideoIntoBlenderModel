"""Tier 1: Algorithmic & Licence Guarantees.

Verifies that the repository, dependency manifests, runtime imports, and asset
stores remain 100% free of copyleft (AGPL) and non-commercial research licences
(such as MPI Body-Model, legacy-lifter, third-party-detector YOLOv8, and legacy legacy_mesh_serializer).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
SOURCE_ROOT = REPO_ROOT / "v2bm"
ASSETS_ROOT = REPO_ROOT / "assets"
WEIGHTS_ROOT = REPO_ROOT / "weights"

PROHIBITED_PACKAGES = {
    "bodymodel",
    "legacy_lifter",
    "third_party_detector",
    "legacy_mesh_serializer",
    "torch",
    "torchvision",
}

APPROVED_PERMISSIVE_PACKAGES = {
    "numpy",
    "opencv-python",
    "mediapipe",
    "scipy",
    "pytest",
    "einops",
    "setuptools",
    "cupy",
    "cupy-cuda12x",
    "cupy-cuda13x",
    "cuda-pathfinder",
}


def test_no_prohibited_dependencies_in_pyproject_toml() -> None:
    """Assert pyproject.toml contains zero copyleft or non-commercial dependencies."""
    assert PYPROJECT_PATH.exists(), f"pyproject.toml missing at {PYPROJECT_PATH}"

    with open(PYPROJECT_PATH, "rb") as f:
        config = tomllib.load(f)

    project = config.get("project", {})
    deps = project.get("dependencies", [])
    optional_deps = project.get("optional-dependencies", {})

    all_dep_strings: list[str] = list(deps)
    for opt_group in optional_deps.values():
        all_dep_strings.extend(opt_group)

    found_violations: list[str] = []
    for dep in all_dep_strings:
        pkg_name = dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        if pkg_name in PROHIBITED_PACKAGES:
            found_violations.append(f"{dep} matches prohibited package '{pkg_name}'")

    assert not found_violations, (
        f"Prohibited non-commercial/copyleft dependencies detected in {PYPROJECT_PATH}:\n"
        + "\n".join(f" - {v}" for v in found_violations)
    )


def test_declared_dependencies_are_permissive() -> None:
    """Verify all dependencies in pyproject.toml belong to approved permissive ecosystems."""
    with open(PYPROJECT_PATH, "rb") as f:
        config = tomllib.load(f)

    project = config.get("project", {})
    deps = project.get("dependencies", [])

    for dep in deps:
        pkg_name = dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        assert pkg_name in APPROVED_PERMISSIVE_PACKAGES, (
            f"Package '{pkg_name}' in pyproject.toml dependencies is not in approved "
            f"permissive list: {sorted(APPROVED_PERMISSIVE_PACKAGES)}"
        )


def test_no_bodymodel_neutral_asset_in_repo() -> None:
    """Ensure neutral_body_model.npz is completely eradicated from assets and weights."""
    forbidden_files = list(REPO_ROOT.glob("**/neutral_body_model.npz")) + list(
        REPO_ROOT.glob("**/*bodymodel*.npz")
    )
    assert not forbidden_files, (
        "Non-commercial MPI Body-Model model file(s) found in repository:\n"
        + "\n".join(f" - {p}" for p in forbidden_files)
    )


def test_downstream_humanoid_modules_strictly_clean() -> None:
    """Verify v2bm/rig/ contains zero prohibited imports."""
    humanoid_dir = SOURCE_ROOT / "rig"
    assert humanoid_dir.is_dir(), f"Expected directory {humanoid_dir}"

    violations: list[str] = []
    for py_file in humanoid_dir.glob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".")[0].lower()
                    if root_name in PROHIBITED_PACKAGES:
                        violations.append(f"{py_file.name}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_name = node.module.split(".")[0].lower()
                if root_name in PROHIBITED_PACKAGES:
                    violations.append(f"{py_file.name}:{node.lineno}: from {node.module} import ...")

    assert not violations, (
        "Downstream humanoid modules must remain 100% clean of prohibited imports:\n"
        + "\n".join(f" - {v}" for v in violations)
    )


def test_checks_and_cli_clean_of_prohibited_imports() -> None:
    """Verify that checks.py and cli.py (Milestone M1 deliverables) contain zero prohibited imports."""
    violations: list[str] = []
    for filename in ("checks.py", "cli.py", "paths.py"):
        py_file = SOURCE_ROOT / filename
        if not py_file.exists():
            continue
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".")[0].lower()
                    if root_name in PROHIBITED_PACKAGES:
                        violations.append(f"{filename}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_name = node.module.split(".")[0].lower()
                if root_name in PROHIBITED_PACKAGES:
                    violations.append(f"{filename}:{node.lineno}: from {node.module} import ...")

    assert not violations, (
        "Milestone M1 files must have zero prohibited imports:\n"
        + "\n".join(f" - {v}" for v in violations)
    )


def test_entire_source_tree_imports_clean() -> None:
    """Audit every Python file under v2bm for copyleft and non-commercial imports."""
    violations: list[str] = []
    for py_file in SOURCE_ROOT.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".")[0].lower()
                    if root_name in PROHIBITED_PACKAGES:
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_name = node.module.split(".")[0].lower()
                if root_name in PROHIBITED_PACKAGES:
                    violations.append(
                        f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: from {node.module} import ..."
                    )

    assert not violations, (
        "Prohibited imports detected across v2bm source tree:\n"
        + "\n".join(f" - {v}" for v in violations)
    )



def test_repository_license_file_permissive() -> None:
    """Verify LICENSE file is Apache-2.0."""
    license_file = REPO_ROOT / "LICENSE"
    assert license_file.exists(), f"LICENSE file missing at {license_file}"
    content = license_file.read_text(encoding="utf-8")
    assert "Apache License" in content and "Version 2.0" in content, (
        "Repository LICENSE must be Apache License 2.0"
    )


def test_no_prohibited_string_literals_in_source() -> None:
    """Enforce Check 1: Zero residual string literals, symbols, or dead definitions in source."""
    prohibited_literals = {
        "BodyModel_NEUTRAL",
        "export_legacy_lifter_env",
        "legacy-lifter_CHECKPOINTS",
        "legacy-lifter_BODY_MODELS",
        "BodyModel_MISSING",
    }
    violations: list[str] = []
    for py_file in SOURCE_ROOT.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for lit in prohibited_literals:
            if lit in text:
                violations.append(f"{py_file.relative_to(REPO_ROOT)}: contains prohibited literal {lit!r}")

    # Assert no leftover bodymodel directory anywhere in repo
    bodymodel_dirs = list(REPO_ROOT.glob("**/bodymodel"))
    for d in bodymodel_dirs:
        violations.append(f"Leftover bodymodel directory found at {d.relative_to(REPO_ROOT)}")

    # Assert no body_models directory
    body_model_dirs = list(REPO_ROOT.glob("**/body_models"))
    for d in body_model_dirs:
        violations.append(f"Leftover body_models directory found at {d.relative_to(REPO_ROOT)}")

    assert not violations, (
        "Prohibited dead-code declarations or residual directories detected:\n"
        + "\n".join(f" - {v}" for v in violations)
    )


def test_licence_registers_exist_and_consistent() -> None:
    """Verify that assets/LICENCES.md and assets/licences.json exist and are consistent."""
    licences_md = ASSETS_ROOT / "LICENCES.md"
    licences_json = ASSETS_ROOT / "licences.json"

    assert licences_md.exists(), f"Missing {licences_md}"
    assert licences_json.exists(), f"Missing {licences_json}"

    with open(licences_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("version") == "1.0.0"
    assets_data = manifest.get("assets", {})
    videos = assets_data.get("videos", [])
    character_rigs = assets_data.get("character_rigs", [])

    assert len(videos) >= 2, f"Expected at least 2 commercial-free videos, got {len(videos)}"
    assert len(character_rigs) >= 2, f"Expected at least 2 character rigs, got {len(character_rigs)}"

    allowed_licences = {
        "CC0-1.0",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "Apache-2.0",
        "MIT",
        "Pexels",
    }

    for item in videos + character_rigs:
        # Check required fields
        assert "title" in item and item["title"], f"Missing title in {item}"
        assert "creator" in item and item["creator"], f"Missing creator in {item}"
        assert "source_url" in item and item["source_url"], f"Missing source_url in {item}"
        assert "local_path" in item and item["local_path"], f"Missing local_path in {item}"
        assert item.get("commercial_use_allowed") is True, f"Commercial use must be allowed: {item}"
        assert item.get("licence_spdx") in allowed_licences, f"Unapproved licence {item.get('licence_spdx')}"

        # Verify physical file existence
        asset_file = REPO_ROOT / item["local_path"]
        assert asset_file.exists(), f"Referenced asset file does not exist on disk: {asset_file}"
        assert asset_file.stat().st_size > 0, f"Asset file is empty: {asset_file}"


def test_sourced_assets_pass_preflight_checks() -> None:
    """Verify all sourced videos and rigs pass v2bm validation checks cleanly."""
    from v2bm import checks

    licences_json = ASSETS_ROOT / "licences.json"
    with open(licences_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assets_data = manifest.get("assets", {})

    for vid in assets_data.get("videos", []):
        vid_path = REPO_ROOT / vid["local_path"]
        err = checks.check_video(vid_path)
        assert err is None, f"check_video failed on {vid_path}: {err}"

    for rig in assets_data.get("character_rigs", []):
        rig_path = REPO_ROOT / rig["local_path"]
        err = checks.check_rig(rig_path)
        assert err is None, f"check_rig failed on {rig_path}: {err}"



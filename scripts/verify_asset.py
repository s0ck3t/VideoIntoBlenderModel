"""Verification script for procedural CC0 Humanoid Humanoid-Rig character asset.

Performs rigorous mathematical and kinematic verification on assets/rigs/humanoid_rig.fbx:
1. Validates binary FBX magic header and checks.check_rig.
2. Extracts core skeleton via fbx_skeleton and verifies all 25 bones.
3. Performs T-pose calibration and checks Det(R_skeleton_to_rig) == 1.0 and forward == [0, 0, 1].
4. Verifies forward kinematics at rest (rest_fk_error <= 1e-12).
5. Verifies linear blend skinning at rest (rest_skin_error < 1e-3 cm).
6. Writes and verifies glTF 2.0 binary (.glb) asset containing meshes and rest animation.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def verify_character_asset(fbx_path: Path, output_glb: Path | None = None) -> bool:
    """Execute all verification stages on the given FBX character asset.

    Parameters
    ----------
    fbx_path : Path
        Path to the character FBX file to verify.
    output_glb : Path | None
        Optional path for the test GLB export. If omitted, a temporary file is used.

    Returns
    -------
    bool
        True if all verification checks pass, False otherwise.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from v2bm.checks import check_rig
    from v2bm.rig.fbx_skeleton import (
        CORE_BONE_SUFFIXES,
        extract_skeleton,
    )
    from v2bm.rig.gltf import _GLTF_MAGIC, write_glb
    from v2bm.rig.retarget import verify_rest_fk
    from v2bm.rig.skinned_mesh import (
        load_character_asset,
        rest_skin_error,
    )
    from v2bm.rig.tpose_calibration import load_calibration

    fbx_path = Path(fbx_path).resolve()
    print("=" * 80)
    print(f"VERIFYING HUMANOID CHARACTER ASSET: {fbx_path}")
    print(f"File size: {fbx_path.stat().st_size:,} bytes")
    print("=" * 80)

    results: list[tuple[str, str, str, bool]] = []

    # -------------------------------------------------------------------------
    # 1. Rig Content Check (checks.check_rig)
    # -------------------------------------------------------------------------
    rig_err = check_rig(fbx_path)
    check1_passed = rig_err is None
    results.append((
        "checks.check_rig",
        "None (Valid Kaydara binary FBX with rig:Hips)",
        "None" if check1_passed else f"Error: {rig_err}",
        check1_passed,
    ))

    # -------------------------------------------------------------------------
    # 2. Skeleton Extraction & Bone Count (fbx_skeleton.extract_skeleton)
    # -------------------------------------------------------------------------
    skel_data = extract_skeleton(fbx_path)
    extracted_bones = [b["name"] for b in skel_data["bones"]]
    expected_count = len(CORE_BONE_SUFFIXES)
    expected_names = {f"rig:{name}" for name in CORE_BONE_SUFFIXES}
    missing_bones = expected_names - set(extracted_bones)
    check2_passed = len(extracted_bones) == expected_count and not missing_bones
    results.append((
        "fbx_skeleton.extract_skeleton",
        f"Exactly {expected_count} core rig:* bones",
        f"{len(extracted_bones)} bones extracted (missing: {sorted(missing_bones) if missing_bones else 'none'})",
        check2_passed,
    ))

    # -------------------------------------------------------------------------
    # 3. T-Pose Calibration (tpose_calibration.load_calibration)
    # -------------------------------------------------------------------------
    calib = load_calibration(fbx_path, verbose=False)
    det_r = float(np.linalg.det(calib.r_skeleton_to_rig))
    det_passed = abs(det_r - 1.0) < 1e-6

    fwd = calib.forward
    target_fwd = np.array([0.0, 0.0, 1.0])
    fwd_passed = bool(np.allclose(fwd, target_fwd, atol=1e-4))

    rig_up = calib.up
    target_up = np.array([0.0, 1.0, 0.0])
    up_passed = bool(np.allclose(rig_up, target_up, atol=1e-4))

    rig_right = calib.right
    target_right = np.array([-1.0, 0.0, 0.0])
    right_passed = bool(np.allclose(rig_right, target_right, atol=1e-4))

    check3_passed = det_passed and fwd_passed and up_passed and right_passed
    results.append((
        "tpose_calibration.load_calibration (Det)",
        "Det(R_skeleton_to_rig) == 1.000000",
        f"{det_r:.6f}",
        det_passed,
    ))
    results.append((
        "tpose_calibration.load_calibration (Axes)",
        "forward=[0, 0, 1], up=[0, 1, 0], right=[-1, 0, 0]",
        f"forward={np.round(fwd, 4).tolist()}, up={np.round(rig_up, 4).tolist()}, right={np.round(rig_right, 4).tolist()}",
        check3_passed,
    ))

    # -------------------------------------------------------------------------
    # 4. Rest Forward Kinematics (retarget.verify_rest_fk)
    # -------------------------------------------------------------------------
    try:
        rest_fk_err = verify_rest_fk(calib.skeleton_rig, atol=1e-12)
        check4_passed = rest_fk_err <= 1e-12
        fk_msg = f"{rest_fk_err:.6e}"
    except Exception as exc:  # noqa: BLE001
        check4_passed = False
        fk_msg = f"Exception: {exc}"
    results.append((
        "retarget.verify_rest_fk",
        "rest_fk_error <= 1.000000e-12",
        fk_msg,
        check4_passed,
    ))

    # -------------------------------------------------------------------------
    # 5. Rest Linear Blend Skinning (skinned_mesh.rest_skin_error)
    # -------------------------------------------------------------------------
    try:
        asset = load_character_asset(fbx_path)
        mesh_names = [m.name for m in asset.meshes]
        has_required_meshes = "Alpha_Surface" in mesh_names and "Alpha_Joints" in mesh_names
        skin_err_m = rest_skin_error(asset)
        skin_err_cm = skin_err_m * 100.0
        check5_passed = skin_err_cm < 1e-3 and has_required_meshes
        skin_msg = f"{skin_err_cm:.6e} cm ({skin_err_m:.6e} m) across meshes {mesh_names}"
    except Exception as exc:  # noqa: BLE001
        check5_passed = False
        skin_msg = f"Exception: {exc}"
    results.append((
        "skinned_mesh.rest_skin_error",
        "rest_skin_error < 1.000000e-03 cm (Alpha_Surface + Alpha_Joints)",
        skin_msg,
        check5_passed,
    ))

    # -------------------------------------------------------------------------
    # 6. glTF 2.0 Export (gltf.write_glb)
    # -------------------------------------------------------------------------
    temp_glb = output_glb or (fbx_path.parent / "temp_verification.glb")
    try:
        r_anim_frames = [{name: np.eye(3) for name in asset.rig.order}]
        hips_cm_frames = [asset.rig.bones["rig:Hips"].rest_local[:3, 3]]
        written_glb = write_glb(temp_glb, asset, r_anim_frames, hips_cm_frames, fps=30.0)

        # Validate binary glTF container header
        with open(written_glb, "rb") as f:
            header = f.read(12)
            magic, version, length = struct.unpack("<III", header)
        glb_size = written_glb.stat().st_size
        glb_valid = magic == _GLTF_MAGIC and version == 2 and length == glb_size
        check6_passed = glb_valid and glb_size > 1000
        glb_msg = f"Valid GLB ({glb_size:,} bytes, magic=0x{magic:08X}, version={version})"
    except Exception as exc:  # noqa: BLE001
        check6_passed = False
        glb_msg = f"Exception: {exc}"
    finally:
        if output_glb is None and temp_glb.exists():
            temp_glb.unlink()
    results.append((
        "gltf.write_glb",
        "Valid binary glTF 2.0 container (.glb)",
        glb_msg,
        check6_passed,
    ))

    # -------------------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"{'CHECK':<32} | {'TARGET / REQUIREMENT':<30} | {'STATUS'}")
    print("-" * 80)
    all_passed = True
    for name, req, measured, ok in results:
        status_str = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"{name:<32} | {req:<30} | {status_str}")
        print(f"  -> Measured: {measured}")

    print("-" * 80)
    overall_status = "ALL VERIFICATION CHECKS PASSED (CC0 ASSET VALIDATED)" if all_passed else "VERIFICATION FAILED"
    print(f"OVERALL RESULT: {overall_status}")
    print("=" * 80)
    return all_passed


def main() -> int:
    """CLI entry point for asset verification."""
    default_fbx = (
        Path(__file__).resolve().parent.parent / "assets" / "rigs" / "humanoid_rig.fbx"
    )
    parser = argparse.ArgumentParser(
        description="Verify Procedural CC0 Humanoid Character Asset"
    )
    parser.add_argument(
        "--fbx",
        type=str,
        default=str(default_fbx),
        help="Path to character FBX asset to verify",
    )
    parser.add_argument(
        "--output-glb",
        type=str,
        default=None,
        help="Optional destination path for test glTF .glb output",
    )
    args = parser.parse_args()

    fbx_file = Path(args.fbx).resolve()
    if not fbx_file.exists():
        print(f"Error: FBX asset not found at {fbx_file}", file=sys.stderr)
        return 1

    out_glb = Path(args.output_glb).resolve() if args.output_glb else None
    passed = verify_character_asset(fbx_file, out_glb)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

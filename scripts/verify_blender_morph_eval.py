"""Headless Blender 5.2.2 LTS Verification Script for GLB Facial Morphs and Action Evaluation.

Executes within the Blender Python environment to verify:
1. Clean glTF 2.0 import with finished status ({'FINISHED'}).
2. Armature presence and 25-bone Humanoid hierarchy.
3. Alpha_Surface / character mesh contains exactly 53 shape keys (Basis + 52 ARKit keys).
4. Active animation data and Action with F-curves driving facial channels (both legacy and Blender 5.x slotted actions).
5. Dynamic timeline mesh geometry evaluation without vertex explosions or NaN/Inf coordinates.
6. Anterior facial vertices (Y >= 158.0 cm) remain rigidly attached to head motion without tearing or shearing.
7. Support for non-morph rig verification (verifying 0 shape keys).

Usage in PowerShell:
  Start-Process -FilePath "$env:LOCALAPPDATA\\Microsoft\\WindowsApps\\blender-launcher.exe" `
    -ArgumentList "--background", "--python", "scripts\\verify_blender_morph_eval.py", "--", `
      "--glb", "<glb_path>", `
      "--json", "<report_path>", `
      "--log", "<log_path>" -Wait
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Canonical 52 ARKit blendshapes
CANONICAL_ARKIT_BLENDSHAPES: list[str] = [
    # Brows (5)
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    # Cheeks (3)
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    # Eyes (14)
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    # Jaw (4)
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    # Mouth (23)
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    # Nose (2)
    "noseSneerLeft",
    "noseSneerRight",
    # Tongue (1)
    "tongueOut",
]

CORE_25_BONES = [
    "rig:Hips",
    "rig:Spine",
    "rig:Spine1",
    "rig:Spine2",
    "rig:Neck",
    "rig:Head",
    "rig:HeadTop_End",
    "rig:LeftShoulder",
    "rig:LeftArm",
    "rig:LeftForeArm",
    "rig:LeftHand",
    "rig:RightShoulder",
    "rig:RightArm",
    "rig:RightForeArm",
    "rig:RightHand",
    "rig:LeftUpLeg",
    "rig:LeftLeg",
    "rig:LeftFoot",
    "rig:LeftToeBase",
    "rig:LeftToe_End",
    "rig:RightUpLeg",
    "rig:RightLeg",
    "rig:RightFoot",
    "rig:RightToeBase",
    "rig:RightToe_End",
]


class DualLogger:
    """Writes logs simultaneously to stdout and a persistent disk log file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1 :]
    else:
        script_args = []

    parser = argparse.ArgumentParser(
        description="Verify glTF 2.0 character animation and morph targets in headless Blender"
    )
    parser.add_argument("--glb", type=Path, required=True, help="Path to .glb file to verify")
    parser.add_argument("--json", type=Path, required=True, help="Path to output structured JSON report")
    parser.add_argument("--log", type=Path, default=None, help="Path to output text log file")
    parser.add_argument(
        "--expect-morphs",
        action="store_true",
        default=True,
        help="Whether morph targets (53 shape keys) are expected",
    )
    parser.add_argument(
        "--no-morphs",
        dest="expect_morphs",
        action="store_false",
        help="Assert that NO shape keys exist (for non-morph rigs)",
    )
    return parser.parse_args(script_args)


def run_evaluation() -> int:
    args = parse_args()
    log_path = args.log or args.json.with_suffix(".log")
    logger = DualLogger(log_path)

    logger.log("=================================================================")
    logger.log("Headless Blender 5.2.2 LTS Morph Target & Action Evaluation")
    logger.log("=================================================================")
    logger.log(f"Target GLB: {args.glb}")
    logger.log(f"Structured JSON: {args.json}")
    logger.log(f"Disk Log: {log_path}")
    logger.log(f"Expect Morphs: {args.expect_morphs}")

    report: dict[str, object] = {
        "status": "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "glb_path": str(args.glb.resolve()) if args.glb.exists() else str(args.glb),
        "expect_morphs": args.expect_morphs,
    }

    try:
        import bpy
        import numpy as np
    except ImportError as e:
        logger.log(f"FATAL: Required module not available: {e}")
        report["error"] = f"Import error: {e}"
        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        logger.close()
        return 1

    blender_version = bpy.app.version_string
    logger.log(f"Blender Version: {blender_version}")
    report["blender_version"] = blender_version

    if not args.glb.exists():
        logger.log(f"FATAL: Target GLB file does not exist: {args.glb}")
        report["error"] = f"File not found: {args.glb}"
        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        logger.close()
        return 1

    try:
        # Step 1: Factory Reset
        logger.log("Step 1: Clearing scene to clean factory empty state...")
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Step 2: Import Scene glTF
        logger.log(f"Step 2: Importing {args.glb} via bpy.ops.import_scene.gltf...")
        import_result = bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))
        logger.log(f"Import operator result: {import_result}")
        assert "FINISHED" in import_result, f"glTF import operator failed: {import_result}"
        report["import_operator_status"] = list(import_result)

        # Step 3: Armature & Bone Hierarchy Validation
        logger.log("Step 3: Validating Armature object and Humanoid bone hierarchy...")
        armature_objs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
        assert len(armature_objs) >= 1, "No ARMATURE object found in imported glTF scene"
        arm_obj = armature_objs[0]
        logger.log(f"Armature object identified: '{arm_obj.name}'")

        bone_names = [b.name for b in arm_obj.data.bones]
        logger.log(f"Total bones in armature: {len(bone_names)}")
        missing_core_bones = []
        for cb in CORE_25_BONES:
            suffix = cb.split(":")[-1]
            if not any(b.endswith(suffix) for b in bone_names):
                missing_core_bones.append(cb)

        assert not missing_core_bones, f"Missing core Humanoid bones: {missing_core_bones}"
        logger.log(f"Armature validation: PASS ({len(bone_names)} bones present, 25 core bones satisfied)")
        report["armature"] = {
            "name": arm_obj.name,
            "bone_count": len(bone_names),
            "missing_core_bones": missing_core_bones,
        }

        # Step 4: Meshes and Shape Keys Inspection
        logger.log("Step 4: Inspecting meshes and shape key data...")
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        assert len(mesh_objs) >= 1, "No MESH objects found in imported glTF scene"
        logger.log(f"Found {len(mesh_objs)} MESH object(s): {[m.name for m in mesh_objs]}")

        # Locate the primary skinned character mesh
        char_mesh_objs = [m for m in mesh_objs if "icosphere" not in m.name.lower()]
        assert char_mesh_objs, "No character skinned mesh object found (excluding helper meshes)"
        char_obj = char_mesh_objs[0]
        logger.log(f"Primary character mesh object: '{char_obj.name}' (data: '{char_obj.data.name}')")

        shape_keys = char_obj.data.shape_keys

        if not args.expect_morphs:
            # Non-morph verification
            logger.log("Step 4 (Non-morph verification): Checking that NO shape keys exist...")
            assert shape_keys is None, (
                f"Expected 0 shape keys on non-morph rig, but found shape_keys block: "
                f"{[kb.name for kb in shape_keys.key_blocks]}"
            )
            logger.log("Non-morph rig shape key validation: PASS (0 shape keys present)")
            report["morph_evaluation"] = {
                "expected": False,
                "shape_keys_present": False,
                "status": "PASS",
            }
            report["status"] = "PASS"
            logger.log("=================================================================")
            logger.log("STATUS: PASS (Non-morph rig glTF verification complete)")
            logger.log("=================================================================")
            with open(args.json, "w", encoding="utf-8") as jf:
                json.dump(report, jf, indent=2)
            logger.close()
            return 0

        # Morph verification
        assert shape_keys is not None, "Mesh object has no shape_keys block, but morphs were expected"
        key_blocks = shape_keys.key_blocks
        key_names = [kb.name for kb in key_blocks]
        logger.log(f"Total shape key blocks: {len(key_names)}")
        logger.log(f"First key block: '{key_names[0]}'")

        # Assert exactly 53 keys: Basis + 52 ARKit keys
        assert len(key_names) == 53, (
            f"Expected exactly 53 shape keys (Basis + 52 ARKit), got {len(key_names)}: {key_names[:10]}..."
        )
        assert key_names[0] == "Basis", f"Expected first key block to be 'Basis', got '{key_names[0]}'"

        morph_keys = key_names[1:]
        missing_arkit = [k for k in CANONICAL_ARKIT_BLENDSHAPES if k not in morph_keys]
        extra_arkit = [k for k in morph_keys if k not in CANONICAL_ARKIT_BLENDSHAPES]

        assert not missing_arkit, f"Missing canonical ARKit blendshapes: {missing_arkit}"
        assert not extra_arkit, f"Unexpected extra blendshapes: {extra_arkit}"
        logger.log("Shape keys 1:1 match with CANONICAL_ARKIT_BLENDSHAPES: PASS (52/52 present)")

        # Step 5: Animation Data and Action F-curves Evaluation
        logger.log("Step 5: Inspecting shape key animation data and Action F-curves...")
        anim_data = shape_keys.animation_data
        assert anim_data is not None, "Shape keys block has no animation_data"
        action = anim_data.action
        assert action is not None, "Shape keys animation_data has no active Action assigned"
        logger.log(f"Active Action on shape keys: '{action.name}'")

        # Extract F-curves across Blender versions (legacy fcurves vs Blender 5.x layered/slotted channelbags)
        fcurves_found: list[str] = []

        # 1. Direct action.fcurves (Blender 4.x / legacy)
        if hasattr(action, "fcurves") and action.fcurves:
            for fc in action.fcurves:
                fcurves_found.append(fc.data_path)

        # 2. Layered action channelbags (Blender 5.x Slotted Actions)
        if hasattr(action, "layers") and action.layers:
            for layer in action.layers:
                for strip in getattr(layer, "strips", []):
                    for cb in getattr(strip, "channelbags", []):
                        for fc in getattr(cb, "fcurves", []):
                            fcurves_found.append(fc.data_path)

        logger.log(f"Total Action F-curves discovered: {len(fcurves_found)}")

        # Check shape key F-curves
        sk_fcurves = [p for p in fcurves_found if "key_blocks" in p]
        logger.log(f"Shape key F-curves discovered: {len(sk_fcurves)}")
        assert len(sk_fcurves) == 52, f"Expected 52 shape key F-curves, found {len(sk_fcurves)}"

        # Assert key facial channels are explicitly driven
        critical_channels = [
            "jawOpen",
            "mouthSmileLeft",
            "mouthSmileRight",
            "eyeBlinkLeft",
            "eyeBlinkRight",
            "browInnerUp",
        ]
        for ch in critical_channels:
            expected_path = f'key_blocks["{ch}"].value'
            assert any(expected_path in p for p in sk_fcurves), f"Critical channel '{ch}' has no active F-curve"

        logger.log(f"Active F-curves confirmed for critical channels: {critical_channels}")

        # Step 6: Step Through Timeline Frames & Evaluate Geometry
        logger.log("Step 6: Stepping through timeline frames and evaluating dynamic deformation...")
        scene = bpy.context.scene
        eval_frames = [0, 5, 8, 10, 12, 15, 20, 25]
        frame_metrics = []
        prev_coords = None

        mesh = char_obj.data
        n_verts = len(mesh.vertices)
        coords_rest = np.empty((n_verts, 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", coords_rest.ravel())

        # Determine vertical height axis in Blender coordinate space
        # In Blender glTF import (+Z up), Z is vertical height (0 to 1.8m)
        height_axis = 2 if coords_rest[:, 2].max() > 1.5 else 1
        logger.log(f"Vertical height axis in Blender: coordinate index {height_axis}")

        for f in eval_frames:
            scene.frame_set(f)
            # Query active shape key values
            jaw_val = float(shape_keys.key_blocks["jawOpen"].value)
            smile_val = float(shape_keys.key_blocks["mouthSmileLeft"].value)
            blink_val = float(shape_keys.key_blocks["eyeBlinkLeft"].value)

            # Evaluate deformed mesh from depsgraph
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_char = char_obj.evaluated_get(depsgraph)
            eval_mesh = eval_char.to_mesh()

            eval_coords = np.empty((len(eval_mesh.vertices), 3), dtype=np.float32)
            eval_mesh.vertices.foreach_get("co", eval_coords.ravel())

            # Numerical stability checks
            assert not np.isnan(eval_coords).any(), f"Frame {f}: NaN coordinates detected in evaluated mesh"
            assert not np.isinf(eval_coords).any(), f"Frame {f}: Infinite coordinates detected in evaluated mesh"

            # Anti-explosion bounds check (humanoid character dimensions should remain within [-3, 3] metres)
            c_min = eval_coords.min(axis=0)
            c_max = eval_coords.max(axis=0)
            span = c_max - c_min
            assert np.all(span < 3.0), f"Frame {f}: Vertex explosion detected! Coordinate span exceeded: {span}"

            delta_prev = 0.0
            if prev_coords is not None:
                delta_prev = float(np.linalg.norm(eval_coords - prev_coords, axis=1).max())

            eval_char.to_mesh_clear()
            prev_coords = eval_coords.copy()

            logger.log(
                f"  Frame {f:2d}: jawOpen={jaw_val:.3f}, mouthSmileLeft={smile_val:.3f}, "
                f"eyeBlinkLeft={blink_val:.3f} | Span=[{span[0]:.2f}, {span[1]:.2f}, {span[2]:.2f}] m, "
                f"Max vertex delta={delta_prev:.4f} m"
            )

            frame_metrics.append({
                "frame": f,
                "jawOpen": jaw_val,
                "mouthSmileLeft": smile_val,
                "eyeBlinkLeft": blink_val,
                "span": [float(v) for v in span],
                "delta_from_previous": delta_prev,
            })

        # Step 7: Verify Rigid Head Attachment & Facial Tearing / Shearing Check
        logger.log("Step 7: Verifying anterior facial vertices attachment and anti-shearing...")

        # Extract Basis rest coordinates
        basis_kb = shape_keys.key_blocks["Basis"]
        basis_co = np.empty((n_verts, 3), dtype=np.float32)
        basis_kb.data.foreach_get("co", basis_co.ravel())

        # Determine all vertices modulated by the 52 ARKit morph targets
        morph_disp_max = np.zeros(n_verts, dtype=np.float32)
        kb_buf = np.empty((n_verts, 3), dtype=np.float32)
        for kb in shape_keys.key_blocks[1:]:
            kb.data.foreach_get("co", kb_buf.ravel())
            disp = np.linalg.norm(kb_buf - basis_co, axis=1)
            morph_disp_max = np.maximum(morph_disp_max, disp)

        # Facial vertices deformed by ARKit morph targets (displacement > 0.5 mm)
        face_vertex_indices = np.where(morph_disp_max > 0.0005)[0]
        logger.log(f"Anterior facial vertices identified via ARKit morph deltas: {len(face_vertex_indices)} vertices")
        assert len(face_vertex_indices) >= 200, (
            f"Expected at least 200 ARKit facial vertices, found {len(face_vertex_indices)}"
        )

        # Coordinate bounding envelope of the active face
        face_coords = basis_co[face_vertex_indices]
        f_min = face_coords.min(axis=0)
        f_max = face_coords.max(axis=0)
        logger.log(
            f"Facial geometry envelope: X=[{f_min[0]:.3f}, {f_max[0]:.3f}] m, "
            f"Y=[{f_min[1]:.3f}, {f_max[1]:.3f}] m, Z=[{f_min[2]:.3f}, {f_max[2]:.3f}] m"
        )
        assert f_min[height_axis] >= 1.57, (
            f"Facial height boundary check failed: minimum height is {f_min[height_axis]:.3f} m (expected >= 1.57 m)"
        )

        # Check bone weight binding
        vg_name_to_idx = {vg.name: vg.index for vg in char_obj.vertex_groups}
        head_vg_idx = None
        for name, idx in vg_name_to_idx.items():
            if name == "rig:Head" or name.endswith(":Head"):
                head_vg_idx = idx
                break

        assert head_vg_idx is not None, "rig:Head vertex group not found on character mesh"

        head_weights = []
        for idx in face_vertex_indices:
            vert = mesh.vertices[int(idx)]
            hw = 0.0
            for g in vert.groups:
                if g.group == head_vg_idx:
                    hw = g.weight
                    break
            head_weights.append(hw)

        head_weights_arr = np.array(head_weights, dtype=np.float32)
        min_hw = float(head_weights_arr.min())
        mean_hw = float(head_weights_arr.mean())
        rigid_count = int(np.count_nonzero(head_weights_arr >= 0.999))
        logger.log(
            f"Anterior face vertex weights to rig:Head: min={min_hw:.4f}, mean={mean_hw:.4f}, "
            f"rigidly bound (100%): {rigid_count}/{len(face_vertex_indices)}"
        )
        assert min_hw >= 0.999, (
            f"Anterior facial vertex not rigidly bound to head bone: min weight is {min_hw:.4f}"
        )
        logger.log("Rigid attachment to rig:Head: PASS (100% of facial vertices have weight 1.0000)")

        # Evaluate edge stretch of facial polygons between rest pose (frame 0) and peak deformation (frame 10)
        scene.frame_set(0)
        deps0 = bpy.context.evaluated_depsgraph_get()
        m0 = char_obj.evaluated_get(deps0).to_mesh()
        c0 = np.empty((len(m0.vertices), 3), dtype=np.float32)
        m0.vertices.foreach_get("co", c0.ravel())
        char_obj.evaluated_get(deps0).to_mesh_clear()

        scene.frame_set(10)
        deps10 = bpy.context.evaluated_depsgraph_get()
        m10 = char_obj.evaluated_get(deps10).to_mesh()
        c10 = np.empty((len(m10.vertices), 3), dtype=np.float32)
        m10.vertices.foreach_get("co", c10.ravel())
        char_obj.evaluated_get(deps10).to_mesh_clear()

        # Check stretch on all facial polygons
        face_idx_set = set(int(v) for v in face_vertex_indices)
        face_edge_stretches = []
        for poly in mesh.polygons:
            poly_verts = list(poly.vertices)
            if any(v in face_idx_set for v in poly_verts):
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i]
                    v2 = poly_verts[(i + 1) % len(poly_verts)]
                    e0 = np.linalg.norm(c0[v1] - c0[v2])
                    e10 = np.linalg.norm(c10[v1] - c10[v2])
                    face_edge_stretches.append(float(abs(e10 - e0)))

        assert face_edge_stretches, "No facial polygon edges evaluated"
        max_stretch_cm = float(max(face_edge_stretches) * 100.0)
        mean_stretch_cm = float((sum(face_edge_stretches) / len(face_edge_stretches)) * 100.0)
        logger.log(
            f"Facial manifold edge stretch between Frame 0 and Frame 10: "
            f"max={max_stretch_cm:.3f} cm, mean={mean_stretch_cm:.3f} cm"
        )

        # Anti-tearing assertion: Maximum edge strain under speech/smile must not exceed 4.0 cm
        assert max_stretch_cm < 4.0, (
            f"Mesh tearing/shearing detected! Maximum facial edge stretch is {max_stretch_cm:.2f} cm (exceeds 4.0 cm limit)"
        )
        logger.log("Anti-tearing and skull attachment validation: PASS (continuous manifold preserved)")

        report["morph_evaluation"] = {
            "expected": True,
            "total_shape_keys": len(key_names),
            "canonical_arkit_count": len(morph_keys),
            "fcurves_count": len(sk_fcurves),
            "timeline_frames_evaluated": len(eval_frames),
            "frame_metrics": frame_metrics,
            "anterior_face_vertex_count": len(face_vertex_indices),
            "head_bone_weight_min": min_hw,
            "head_bone_weight_mean": mean_hw,
            "max_edge_stretch_cm": max_stretch_cm,
            "mean_edge_stretch_cm": mean_stretch_cm,
            "status": "PASS",
        }

        # Step 8: Final Pass
        report["status"] = "PASS"
        logger.log("=================================================================")
        logger.log("STATUS: PASS (All Headless Blender LTS evaluations succeeded)")
        logger.log("=================================================================")

        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)

        logger.close()
        return 0

    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        tb_str = traceback.format_exc()
        logger.log("-----------------------------------------------------------------")
        logger.log(f"VERIFICATION FAILURE: {err_msg}")
        logger.log(tb_str)
        logger.log("STATUS: FAIL")
        logger.log("-----------------------------------------------------------------")

        report["status"] = "FAIL"
        report["error"] = err_msg
        report["traceback"] = tb_str

        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)

        logger.close()
        return 1


if __name__ == "__main__":
    sys.exit(run_evaluation())

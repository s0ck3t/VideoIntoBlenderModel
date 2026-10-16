"""Tier 2: Mathematical Verification — Forward Kinematics & Linear Blend Skinning.

Verifies:
- rest_fk_error <= 1e-12 (Machine precision identity forward kinematics)
- rest_skin_error_cm < 1e-3 cm (Sub-micrometre linear blend skinning fidelity at rest)
- Bone hierarchy and inverse bind matrix consistency
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v2bm.rig.fbx_skeleton import CORE_BONE_SUFFIXES
from v2bm.rig.kinematics import (
    RigBone,
    HumanoidRig,
    forward_kinematics,
)
from v2bm.rig.retarget import verify_rest_fk
from v2bm.rig.skinned_mesh import (
    SkinnedMesh,
    HumanoidRigAsset,
    load_character_asset,
    rest_skin_error,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FBX_ASSET_PATH = REPO_ROOT / "assets" / "rigs" / "humanoid_rig.fbx"


def create_synthetic_humanoid_rig() -> HumanoidRig:
    """Construct an in-memory 25-bone HumanoidRig with identity rest rotations."""
    # Hierarchy definition (head positions in cm)
    bone_defs: list[tuple[str, str | None, tuple[float, float, float]]] = [
        ("rig:Hips", None, (0.0, 102.0, 0.0)),
        ("rig:Spine", "rig:Hips", (0.0, 112.0, 0.0)),
        ("rig:Spine1", "rig:Spine", (0.0, 124.0, 0.0)),
        ("rig:Spine2", "rig:Spine1", (0.0, 136.0, 0.0)),
        ("rig:Neck", "rig:Spine2", (0.0, 150.0, 0.0)),
        ("rig:Head", "rig:Neck", (0.0, 160.0, 0.0)),
        ("rig:HeadTop_End", "rig:Head", (0.0, 175.0, 0.0)),
        ("rig:LeftShoulder", "rig:Spine2", (5.0, 148.0, 0.0)),
        ("rig:LeftArm", "rig:LeftShoulder", (18.0, 148.0, 0.0)),
        ("rig:LeftForeArm", "rig:LeftArm", (45.0, 148.0, 0.0)),
        ("rig:LeftHand", "rig:LeftForeArm", (72.0, 148.0, 0.0)),
        ("rig:RightShoulder", "rig:Spine2", (-5.0, 148.0, 0.0)),
        ("rig:RightArm", "rig:RightShoulder", (-18.0, 148.0, 0.0)),
        ("rig:RightForeArm", "rig:RightArm", (-45.0, 148.0, 0.0)),
        ("rig:RightHand", "rig:RightForeArm", (-72.0, 148.0, 0.0)),
        ("rig:LeftUpLeg", "rig:Hips", (10.0, 96.0, 0.0)),
        ("rig:LeftLeg", "rig:LeftUpLeg", (10.0, 52.0, 0.0)),
        ("rig:LeftFoot", "rig:LeftLeg", (10.0, 10.0, 0.0)),
        ("rig:LeftToeBase", "rig:LeftFoot", (10.0, 3.0, 14.0)),
        ("rig:LeftToe_End", "rig:LeftToeBase", (10.0, 3.0, 22.0)),
        ("rig:RightUpLeg", "rig:Hips", (-10.0, 96.0, 0.0)),
        ("rig:RightLeg", "rig:RightUpLeg", (-10.0, 52.0, 0.0)),
        ("rig:RightFoot", "rig:RightLeg", (-10.0, 10.0, 0.0)),
        ("rig:RightToeBase", "rig:RightFoot", (-10.0, 3.0, 14.0)),
        ("rig:RightToe_End", "rig:RightToeBase", (-10.0, 3.0, 22.0)),
    ]

    bones: dict[str, RigBone] = {}
    order: list[str] = [name for name, _, _ in bone_defs]
    head_pos: dict[str, np.ndarray] = {name: np.array(pos, dtype=np.float64) for name, _, pos in bone_defs}

    for name, parent, _ in bone_defs:
        children = [child_name for child_name, p, _ in bone_defs if p == name]
        if parent is None:
            lcl_t = head_pos[name].copy()
        else:
            lcl_t = head_pos[name] - head_pos[parent]

        rest_loc = np.eye(4, dtype=np.float64)
        rest_loc[:3, 3] = lcl_t

        rest_w = np.eye(4, dtype=np.float64)
        rest_w[:3, 3] = head_pos[name]

        bones[name] = RigBone(
            name=name,
            parent=parent,
            children=children,
            pre_rotation=np.eye(3, dtype=np.float64),
            lcl_translation_cm=lcl_t,
            rest_world=rest_w,
            rest_local=rest_loc,
            pre_rotation_euler_deg=np.zeros(3, dtype=np.float64),
        )

    return HumanoidRig(
        bones=bones,
        order=order,
        meters_per_unit=0.01,
        source=Path("synthetic_rig"),
        coord_system={"fbx_unit": "centimeter", "meters_per_fbx_unit": 0.01},
    )


def test_in_memory_fk_rest_error_machine_precision() -> None:
    """Verify forward kinematics achieves rest_fk_error <= 1e-12 on an ideal rig."""
    rig = create_synthetic_humanoid_rig()
    err = verify_rest_fk(rig, atol=1e-12)
    assert err <= 1e-12, f"Expected rest_fk_error <= 1e-12, got {err:.3e}"


def test_in_memory_skinning_rest_error_sub_millimetre() -> None:
    """Verify linear blend skinning achieves rest_skin_error < 1e-3 cm at rest."""
    rig = create_synthetic_humanoid_rig()

    # Generate synthetic vertices near each bone
    vertices_list = []
    n_bones = len(rig.order)
    weights_list = []

    for idx, (name, bone) in enumerate(rig.bones.items()):
        pos = bone.rest_world[:3, 3]
        # 4 vertices per joint
        for offset in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]:
            v = pos + np.array(offset, dtype=np.float64)
            vertices_list.append(v)
            w = np.zeros(n_bones, dtype=np.float64)
            w[idx] = 1.0
            weights_list.append(w)

    vertices = np.array(vertices_list, dtype=np.float64)
    weights = np.array(weights_list, dtype=np.float64)

    # Inverse bind matrices: for each bone, IBM = inv(rest_world)
    ibm = np.stack([np.linalg.inv(rig.bones[name].rest_world) for name in rig.order], axis=0)

    dummy_faces = np.zeros((0, 3), dtype=np.int32)
    mesh = SkinnedMesh(
        name="Alpha_Surface",
        vertices=vertices,
        faces=dummy_faces,
        weights=weights,
        ibm=ibm,
        color=np.array([0.6, 0.6, 0.6]),
        face_colors=None,
    )

    asset = HumanoidRigAsset(
        rig=rig,
        meshes=[mesh],
        bone_index={name: i for i, name in enumerate(rig.order)},
    )
    skin_err = rest_skin_error(asset)
    assert skin_err < 1e-3, f"Expected rest_skin_error < 1e-3 cm, got {skin_err:.6e} cm"



def test_forward_kinematics_hips_translation_linearity() -> None:
    """Test that rigid root translations shift world positions uniformly without distortion."""
    rig = create_synthetic_humanoid_rig()
    delta_t = np.array([12.5, -7.0, 35.2], dtype=np.float64)

    world_base = forward_kinematics(rig, {}, hips_translation_cm=np.zeros(3))
    world_trans = forward_kinematics(rig, {}, hips_translation_cm=delta_t)

    for name in rig.order:
        p_base = world_base[name][:3, 3]
        p_trans = world_trans[name][:3, 3]
        diff = np.linalg.norm((p_trans - p_base) - delta_t)
        assert diff <= 1e-12, f"Bone '{name}' translation deviated by {diff:.3e} cm"


@pytest.mark.skipif(not FBX_ASSET_PATH.exists(), reason="assets/rigs/humanoid_rig.fbx pending generation")
def test_humanoid_rig_fbx_asset_mathematical_guarantees() -> None:
    """Rigorous audit of the generated CC0 Humanoid humanoid_rig.fbx asset."""
    asset = load_character_asset(FBX_ASSET_PATH)

    # Check 1: 25 core bones
    assert len(asset.rig.bones) == 25, f"Expected 25 bones, got {len(asset.rig.bones)}"
    for suffix in CORE_BONE_SUFFIXES:
        expected_name = f"rig:{suffix}"
        assert expected_name in asset.rig.bones, f"Missing required core bone: {expected_name}"

    # Check 2: rest_fk_error <= 1e-12
    fk_err = verify_rest_fk(asset.rig, atol=1e-12)
    assert fk_err <= 1e-12, f"humanoid_rig.fbx rest FK error exceeded tolerance: {fk_err:.3e} > 1e-12"

    # Check 3: rest_skin_error < 1e-3 cm
    skin_err = rest_skin_error(asset)
    assert skin_err < 1e-3, f"humanoid_rig.fbx rest skinning error exceeded tolerance: {skin_err:.6e} cm"

    # Check 4: Mesh naming and vertex groups
    mesh_names = [m.name for m in asset.meshes]
    assert "Alpha_Surface" in mesh_names, f"Expected 'Alpha_Surface' mesh, got {mesh_names}"
    assert "Alpha_Joints" in mesh_names, f"Expected 'Alpha_Joints' mesh, got {mesh_names}"

    # Check 5: Partition of unity on skin weights
    for mesh in asset.meshes:
        weight_sums = mesh.weights.sum(axis=1)
        max_deviation = float(np.max(np.abs(weight_sums - 1.0)))
        assert max_deviation < 1e-5, (
            f"Mesh '{mesh.name}' skin weights violate partition of unity: max deviation {max_deviation:.3e}"
        )

    # Check 6: Embedded studio gradient textures
    asset_tex = load_character_asset(FBX_ASSET_PATH, textures=True)
    for mesh in asset_tex.meshes:
        assert mesh.face_colors is not None, (
            f"Mesh '{mesh.name}' failed to load embedded studio gradient face colours"
        )
        assert mesh.face_colors.shape == (len(mesh.faces), 3), (
            f"Mesh '{mesh.name}' face colour shape mismatch: expected {(len(mesh.faces), 3)}, got {mesh.face_colors.shape}"
        )
        assert np.all(np.isfinite(mesh.face_colors)), (
            f"Mesh '{mesh.name}' face colours contain non-finite values"
        )

    # Check 7: Exactly zero cross-limb non-adjacent triangle bindings
    parent_child: set[tuple[str, str]] = set()
    for bname, bone in asset.rig.bones.items():
        if bone.parent:
            parent_child.add((bone.parent, bname))
            parent_child.add((bname, bone.parent))

    for mesh in asset.meshes:
        cross_count = 0
        for tri in mesh.faces:
            used_bones: set[str] = set()
            for v_idx in tri:
                for b_idx in np.where(mesh.weights[v_idx] > 1e-3)[0]:
                    used_bones.add(asset.rig.order[b_idx])
            used_list = list(used_bones)
            for i in range(len(used_list)):
                for j in range(i + 1, len(used_list)):
                    b1, b2 = used_list[i], used_list[j]
                    if b1 != b2 and (b1, b2) not in parent_child:
                        cross_count += 1
                        break
        assert cross_count == 0, (
            f"Mesh '{mesh.name}' has {cross_count} cross-limb non-adjacent triangle bindings!"
        )

    # Check 8: Exactly 1 continuous connected component in Alpha_Surface (zero disconnected/floating pieces)
    surface_mesh = [m for m in asset.meshes if m.name == "Alpha_Surface"][0]
    adj: dict[int, list[int]] = {i: [] for i in range(len(surface_mesh.vertices))}
    for tri in surface_mesh.faces:
        adj[tri[0]].extend([tri[1], tri[2]])
        adj[tri[1]].extend([tri[0], tri[2]])
        adj[tri[2]].extend([tri[0], tri[1]])

    visited: set[int] = set()
    components = 0
    for v_i in range(len(surface_mesh.vertices)):
        if v_i not in visited:
            components += 1
            stack = [v_i]
            visited.add(v_i)
            while stack:
                curr = stack.pop()
                for neighbour in adj[curr]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)
    assert components == 1, (
        f"Mesh 'Alpha_Surface' must be a single connected organic manifold, but has {components} disconnected components!"
    )


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_to_matrix(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    c1 = 1.0 - c
    x, y, z = ax
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


@pytest.mark.skipif(not FBX_ASSET_PATH.exists(), reason="assets/rigs/humanoid_rig.fbx pending generation")
def test_deformed_pose_deformation_bounds() -> None:
    """Verify that linear blend skinning under severe deformation remains strictly bounded (edge <= 15.0 cm)."""
    from v2bm.rig.skinned_mesh import posed_vertices

    asset = load_character_asset(FBX_ASSET_PATH)

    # 1. Test canonical synthetic articulation poses (squat, high reach, torso twist)
    test_poses: list[dict[str, np.ndarray]] = [
        # Pose A: Arms hanging down, head tilted
        {
            "rig:LeftArm": _axis_angle_to_matrix((0, 0, 1), -np.radians(75.0)),
            "rig:RightArm": _axis_angle_to_matrix((0, 0, 1), np.radians(75.0)),
            "rig:Head": _axis_angle_to_matrix((1, 0, 0), np.radians(30.0)),
        },
        # Pose B: Deep squat (hips lowered, thighs flexed 90 deg, knees bent 90 deg)
        {
            "rig:LeftUpLeg": _axis_angle_to_matrix((1, 0, 0), np.radians(90.0)),
            "rig:RightUpLeg": _axis_angle_to_matrix((1, 0, 0), np.radians(90.0)),
            "rig:LeftLeg": _axis_angle_to_matrix((1, 0, 0), -np.radians(90.0)),
            "rig:RightLeg": _axis_angle_to_matrix((1, 0, 0), -np.radians(90.0)),
            "rig:Spine": _axis_angle_to_matrix((1, 0, 0), np.radians(25.0)),
            "rig:Spine1": _axis_angle_to_matrix((1, 0, 0), np.radians(25.0)),
        },
        # Pose C: Asymmetric martial arts kick & torso twist
        {
            "rig:LeftUpLeg": _axis_angle_to_matrix((1, 0, 0), np.radians(110.0)),
            "rig:LeftLeg": _axis_angle_to_matrix((1, 0, 0), -np.radians(45.0)),
            "rig:Spine2": _axis_angle_to_matrix((0, 1, 0), np.radians(45.0)),
            "rig:LeftArm": _axis_angle_to_matrix((0, 1, 0), np.radians(60.0)),
            "rig:RightArm": _axis_angle_to_matrix((0, 0, 1), np.radians(90.0)),
        },
    ]

    for pose_idx, r_anim in enumerate(test_poses):
        p_meshes = posed_vertices(asset, r_anim, hips_translation_cm=np.zeros(3))
        for m_idx, mesh in enumerate(asset.meshes):
            p_verts = p_meshes[m_idx]
            e0 = np.linalg.norm(p_verts[mesh.faces[:, 1]] - p_verts[mesh.faces[:, 0]], axis=1)
            e1 = np.linalg.norm(p_verts[mesh.faces[:, 2]] - p_verts[mesh.faces[:, 1]], axis=1)
            e2 = np.linalg.norm(p_verts[mesh.faces[:, 0]] - p_verts[mesh.faces[:, 2]], axis=1)
            max_edge = float(max(e0.max(), e1.max(), e2.max()))
            assert max_edge <= 15.0, (
                f"Synthetic pose {pose_idx} on mesh '{mesh.name}' exceeded maximum edge length: {max_edge:.2f} cm > 15.0 cm"
            )

    # 2. Test full 256-frame dance sequence if output animation is present
    dance_anim_path = REPO_ROOT / "outputs" / "20261014_135759_10264814-uhd_4096_2160_25fps" / "rig_rotations.npz"
    if dance_anim_path.exists():
        data = np.load(dance_anim_path, allow_pickle=True)
        bone_names = [str(n) for n in data["bone_names"]]
        quats = data["quat_wxyz_rig"]
        hips_cm = data["hips_translation_cm_rig"]
        n_frames = len(quats)

        for m_idx, mesh in enumerate(asset.meshes):
            e0_idx = mesh.faces[:, 0]
            e1_idx = mesh.faces[:, 1]
            e2_idx = mesh.faces[:, 2]

            # Rest edge lengths
            r0 = np.linalg.norm(mesh.vertices[e1_idx] - mesh.vertices[e0_idx], axis=1)
            r1 = np.linalg.norm(mesh.vertices[e2_idx] - mesh.vertices[e1_idx], axis=1)
            r2 = np.linalg.norm(mesh.vertices[e0_idx] - mesh.vertices[e2_idx], axis=1)
            rest_edges = np.concatenate([r0, r1, r2])

            max_edge_all = 0.0
            max_p99_ratio = 0.0
            max_ratio_all = 0.0
            for f_idx in range(n_frames):
                r_frame = {bone_names[b]: _quat_wxyz_to_matrix(quats[f_idx, b]) for b in range(len(bone_names))}
                p_verts = posed_vertices(asset, r_frame, hips_translation_cm=hips_cm[f_idx])[m_idx]

                e0_p = np.linalg.norm(p_verts[e1_idx] - p_verts[e0_idx], axis=1)
                e1_p = np.linalg.norm(p_verts[e2_idx] - p_verts[e1_idx], axis=1)
                e2_p = np.linalg.norm(p_verts[e0_idx] - p_verts[e2_idx], axis=1)
                deformed_edges = np.concatenate([e0_p, e1_p, e2_p])
                max_edge_all = max(max_edge_all, float(deformed_edges.max()))

                ratios = deformed_edges / np.maximum(rest_edges, 1e-6)
                max_p99_ratio = max(max_p99_ratio, float(np.percentile(ratios, 99)))
                max_ratio_all = max(max_ratio_all, float(np.max(ratios)))

            assert max_edge_all <= 15.0, (
                f"Dance animation on mesh '{mesh.name}' exceeded maximum edge length: {max_edge_all:.2f} cm > 15.0 cm"
            )
            # Mathematical stretch ratio bound (preventing surface tearing and explosive stretch)
            assert max_p99_ratio <= 3.0, (
                f"Dance animation on mesh '{mesh.name}' exceeded maximum p99 stretch ratio: {max_p99_ratio:.2f} > 3.0"
            )
            assert max_ratio_all <= 8.0, (
                f"Dance animation on mesh '{mesh.name}' exceeded maximum edge stretch ratio: {max_ratio_all:.2f} > 8.0"
            )


def test_sourced_open_licence_rigs_mathematical_guarantees() -> None:
    """Verify that both third-party open-licence rigs satisfy rigorous mathematical and anatomical guarantees.

    Guarantees:
    - Rest forward kinematics error < 1e-4 (machine precision identity)
    - Rest linear blend skinning error < 1e-2 cm
    - At least 25 core humanoid bones
    - Anatomical rest bone hierarchy (legs descend monotonically, arms extend laterally, non-collapsed segments)
    - Non-identity articulated poses deform cleanly without mesh explosion or tearing
    """
    from scipy.spatial.transform import Rotation
    from v2bm.rig.tpose_calibration import load_calibration

    rig_paths = [
        REPO_ROOT / "assets" / "rigs" / "kenney_skater.fbx",
        REPO_ROOT / "assets" / "rigs" / "kenney_female.fbx",
    ]

    for rig_path in rig_paths:
        assert rig_path.exists(), f"Rig missing at {rig_path}"

        cal = load_calibration(rig_path)
        skel = cal.skeleton_rig
        fk_err = verify_rest_fk(skel)
        assert fk_err < 1e-4, f"Rest FK error {fk_err:.4e} exceeds 1e-4 on {rig_path.name}"

        asset = load_character_asset(rig_path, textures=True)
        skin_err = rest_skin_error(asset)
        assert skin_err < 1e-2, (
            f"Rest skin error {skin_err:.4e} cm exceeds 1e-2 cm on {rig_path.name}"
        )

        assert len(skel.order) >= 25, (
            f"Expected at least 25 bones, got {len(skel.order)} on {rig_path.name}"
        )

        # 1. Anatomical vertical hierarchy (cm)
        pos = {name.split(":")[-1]: skel.bones[name].rest_world[:3, 3] for name in skel.order}
        assert pos["Head"][1] > pos["Neck"][1] > pos["Spine2"][1] > pos["Spine1"][1] > pos["Spine"][1] > pos["Hips"][1], (
            f"Corrupted spine vertical hierarchy on {rig_path.name}: {[ (k, float(pos[k][1])) for k in ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head') ]}"
        )
        for side in ("Left", "Right"):
            assert pos["Hips"][1] > pos[f"{side}Leg"][1] > pos[f"{side}Foot"][1] > pos[f"{side}ToeBase"][1], (
                f"Corrupted {side} leg vertical hierarchy on {rig_path.name}: "
                f"Hips={pos['Hips'][1]:.2f}, Leg={pos[f'{side}Leg'][1]:.2f}, Foot={pos[f'{side}Foot'][1]:.2f}, Toe={pos[f'{side}ToeBase'][1]:.2f}"
            )

        # 2. Anatomical lateral arm hierarchy (cm)
        assert pos["LeftShoulder"][0] < pos["LeftArm"][0] < pos["LeftForeArm"][0] < pos["LeftHand"][0], (
            f"Corrupted Left arm lateral hierarchy on {rig_path.name}: "
            f"{[ (k, float(pos[k][0])) for k in ('LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand') ]}"
        )
        assert pos["RightShoulder"][0] > pos["RightArm"][0] > pos["RightForeArm"][0] > pos["RightHand"][0], (
            f"Corrupted Right arm lateral hierarchy on {rig_path.name}: "
            f"{[ (k, float(pos[k][0])) for k in ('RightShoulder', 'RightArm', 'RightForeArm', 'RightHand') ]}"
        )

        # 3. Non-collapsed limb segment lengths (>= 8.0 cm)
        limb_segments = [
            ("LeftUpLeg", "LeftLeg"),
            ("LeftLeg", "LeftFoot"),
            ("RightUpLeg", "RightLeg"),
            ("RightLeg", "RightFoot"),
            ("LeftArm", "LeftForeArm"),
            ("LeftForeArm", "LeftHand"),
            ("RightArm", "RightForeArm"),
            ("RightForeArm", "RightHand"),
        ]
        for b_parent, b_child in limb_segments:
            seg_len = float(np.linalg.norm(pos[b_child] - pos[b_parent]))
            assert seg_len >= 8.0, (
                f"Collapsed limb segment {b_parent}->{b_child} ({seg_len:.2f} cm < 8.0 cm) on {rig_path.name}"
            )

        # 4. Non-identity articulated deformation check
        from v2bm.rig.skinned_mesh import posed_vertices
        r_articulated = {
            "rig:LeftArm": Rotation.from_euler("z", -35.0, degrees=True).as_matrix(),
            "rig:RightArm": Rotation.from_euler("z", 35.0, degrees=True).as_matrix(),
            "rig:LeftUpLeg": Rotation.from_euler("x", -40.0, degrees=True).as_matrix(),
            "rig:LeftLeg": Rotation.from_euler("x", 50.0, degrees=True).as_matrix(),
            "rig:RightUpLeg": Rotation.from_euler("x", 25.0, degrees=True).as_matrix(),
        }
        deformed_meshes = posed_vertices(asset, r_articulated, hips_translation_cm=None)
        for m_idx, mesh in enumerate(asset.meshes):
            r_verts = mesh.vertices
            p_verts = deformed_meshes[m_idx]
            assert np.all(np.isfinite(p_verts)), f"Non-finite vertices under articulated pose on {rig_path.name}"
            # Ensure vertices actually moved under non-identity rotation
            max_disp = float(np.linalg.norm(p_verts - r_verts, axis=1).max())
            assert max_disp > 5.0, f"Articulated pose failed to move mesh vertices on {rig_path.name}"

            e0_idx = mesh.faces[:, 0]
            e1_idx = mesh.faces[:, 1]
            e2_idx = mesh.faces[:, 2]
            rest_edges = np.concatenate([
                np.linalg.norm(r_verts[e1_idx] - r_verts[e0_idx], axis=1),
                np.linalg.norm(r_verts[e2_idx] - r_verts[e1_idx], axis=1),
                np.linalg.norm(r_verts[e0_idx] - r_verts[e2_idx], axis=1),
            ])
            def_edges = np.concatenate([
                np.linalg.norm(p_verts[e1_idx] - p_verts[e0_idx], axis=1),
                np.linalg.norm(p_verts[e2_idx] - p_verts[e1_idx], axis=1),
                np.linalg.norm(p_verts[e0_idx] - p_verts[e2_idx], axis=1),
            ])
            ratios = def_edges / np.maximum(rest_edges, 1e-6)
            p99_ratio = float(np.percentile(ratios, 99))
            assert p99_ratio <= 2.5, (
                f"Articulated pose caused excessive triangle stretch (p99={p99_ratio:.2f} > 2.5) on {rig_path.name}"
            )


def test_perspective_camera_solver_avoids_limit_cycle_explosion() -> None:
    """Ensure build_perspective_camera stays bounded even when trajectory depth varies across several metres."""
    from v2bm.rig.render import build_perspective_camera

    rng = np.random.default_rng(42)
    # Simulate 60 frames where the subject moves across Z in [-2.5, +2.5] m with height [0.0, 1.7] m
    pts = rng.uniform(low=[-0.8, 0.0, -2.5], high=[0.8, 1.7, 2.5], size=(60, 25, 3))
    axes = {
        "up": np.array([0.0, 1.0, 0.0]),
        "forward": np.array([0.0, 0.0, 1.0]),
        "right": np.array([1.0, 0.0, 0.0]),
    }
    cam = build_perspective_camera(axes, pts, width=640, height=640)
    dist = float(np.linalg.norm(cam.eye))
    assert 2.0 <= dist <= 25.0, f"Camera distance {dist:.2f} m out of reasonable bounds [2.0, 25.0]"
    # Project points and confirm none cross behind near plane
    depths = cam.depth(pts.reshape(-1, 3))
    assert np.all(depths > cam.near), "Some trajectory points fell behind the camera near plane"

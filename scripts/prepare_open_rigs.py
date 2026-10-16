"""Utility script to prepare and normalise open-licence character rigs.

Imports third-party CC0 character models in headless Blender 5.2.2 LTS,
prunes auxiliary control bones, sets uniform +Y edit bone tail orientation,
applies rest pose, embeds diffuse textures, and exports binary FBX character rigs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BLENDER_LAUNCHER = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Microsoft"
    / "WindowsApps"
    / "blender-launcher.exe"
)


def prepare_rig_in_blender(
    fbx_in: Path,
    tex_in: Path,
    fbx_out: Path,
    mat_name: str = "CharacterMaterial",
) -> None:
    """Generate and export a clean binary FBX rig with embedded textures via Blender."""
    script_content = f"""
import math
import sys
import traceback
from pathlib import Path
import bpy

try:
    fbx_in = r"{fbx_in.resolve().as_posix()}"
    tex_in = r"{tex_in.resolve().as_posix()}"
    fbx_out = r"{fbx_out.resolve().as_posix()}"

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_in)

    arm_obj = None
    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            arm_obj = obj
        elif obj.type == "MESH":
            mesh_obj = obj

    if not arm_obj or not mesh_obj:
        raise RuntimeError("Failed to identify Armature and Mesh objects in imported scene")

    def clean_kenney_rig(arm, mesh, target_height=175.0):
        # Unparent mesh temporarily before applying transforms to avoid double-scaling
        matrix_world = mesh.matrix_world.copy()
        mesh.parent = None
        mesh.matrix_world = matrix_world

        current_height = mesh.dimensions.z
        if current_height < 1e-4:
            current_height = max(mesh.dimensions.y, mesh.dimensions.z)
        scale_factor = target_height / current_height

        # Rotate around X by -90 deg so Z-up transforms into canonical +Y-up
        arm.rotation_euler = (math.radians(-90.0), 0.0, 0.0)
        mesh.rotation_euler = (math.radians(-90.0), 0.0, 0.0)
        arm.scale = (scale_factor, scale_factor, scale_factor)
        mesh.scale = (scale_factor, scale_factor, scale_factor)

        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        mesh.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        mesh.parent = arm

        # Bake rest pose to establish neutral stance
        bpy.ops.object.mode_set(mode="POSE")
        bpy.ops.pose.select_all(action="SELECT")
        bpy.ops.pose.armature_apply(selected=False)

        # In Edit mode: prune auxiliary non-deforming control bones
        bpy.ops.object.mode_set(mode="EDIT")
        bones_to_delete = [
            eb.name for eb in arm.data.edit_bones
            if any(tag in eb.name for tag in ["Ctrl", "Roll", "IK"])
        ]
        for bname in bones_to_delete:
            eb = arm.data.edit_bones.get(bname)
            if eb:
                arm.data.edit_bones.remove(eb)

        # Unparent Hips to establish canonical skeletal root
        hips = arm.data.edit_bones.get("Hips")
        if hips:
            hips.parent = None

        # Disconnect all edit bones first so setting parent tails never shifts child heads
        for eb in arm.data.edit_bones:
            eb.use_connect = False

        # Establish uniform +Y edit bone tail orientation (5.0 cm tail length for 175 cm human scale)
        for eb in arm.data.edit_bones:
            eb.tail = (eb.head[0], eb.head[1] + 5.0, eb.head[2])
            eb.roll = 0.0

        bpy.ops.object.mode_set(mode="OBJECT")

    # Apply canonical Kenney rig cleaning, transformation, and bone orientation
    clean_kenney_rig(arm_obj, mesh_obj, target_height=175.0)

    # Configure PBR material and embed diffuse texture map
    mat = bpy.data.materials.new(name="{mat_name}")
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf:
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

    img = bpy.data.images.load(tex_in)
    img.pack()
    tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    # Export binary FBX using standard project parameters
    bpy.ops.export_scene.fbx(
        filepath=fbx_out,
        check_existing=False,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Z",
        axis_up="Y",
        object_types={{"ARMATURE", "MESH"}},
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="NULL",
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
    )
    print("STATUS: SUCCESS")
except Exception as exc:
    print(f"ERROR: {{exc}}")
    traceback.print_exc()
    sys.exit(1)
"""
    tmp_script = REPO_ROOT / ".agents" / "teamwork" / "worker_m4_remediation" / f"prepare_{fbx_out.stem}.py"
    tmp_script.parent.mkdir(parents=True, exist_ok=True)
    tmp_script.write_text(script_content, encoding="utf-8")

    cmd = [
        "powershell",
        "-Command",
        (
            f"Start-Process -FilePath '{BLENDER_LAUNCHER}' "
            f"-ArgumentList '--background', '--python', '{tmp_script.resolve()}' -Wait"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Blender execution failed: {res.stderr}")


def main() -> None:
    """Prepare and normalise both commercial-free character rigs."""
    rigs_dir = REPO_ROOT / "assets" / "rigs"
    rigs_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        (
            "kenney_skater",
            REPO_ROOT / "scratch" / "kenney" / "protagonists" / "Model" / "characterMedium.fbx",
            REPO_ROOT / "scratch" / "kenney" / "protagonists" / "Skins" / "skaterMaleA.png",
            "SkaterMaterial",
        ),
        (
            "kenney_female",
            REPO_ROOT / "scratch" / "kenney" / "retro" / "Model" / "characterMedium.fbx",
            REPO_ROOT / "scratch" / "kenney" / "retro" / "Skins" / "humanFemaleA.png",
            "FemaleMaterial",
        ),
    ]

    for name, fbx_src, tex_src, mat_name in sources:
        raw_backup = REPO_ROOT / ".agents" / "teamwork" / "worker_m4_remediation" / "raw_assets" / f"{name}_raw.fbx"
        if fbx_src.exists():
            resolved_fbx_in = fbx_src
        elif raw_backup.exists():
            resolved_fbx_in = raw_backup
        else:
            resolved_fbx_in = rigs_dir / f"{name}.fbx"

        if tex_src.exists():
            resolved_tex_in = tex_src
        else:
            resolved_tex_in = rigs_dir / f"{name}.png"

        temp_fbx_dst = rigs_dir / f"{name}_temp.fbx"
        final_fbx_dst = rigs_dir / f"{name}.fbx"
        final_tex_dst = rigs_dir / f"{name}.png"

        print(f"Preparing {name} from {resolved_fbx_in} -> {final_fbx_dst}...")
        prepare_rig_in_blender(resolved_fbx_in, resolved_tex_in, temp_fbx_dst, mat_name=mat_name)

        if not temp_fbx_dst.exists() or temp_fbx_dst.stat().st_size == 0:
            raise RuntimeError(f"Failed to generate {temp_fbx_dst}")

        shutil.move(str(temp_fbx_dst), str(final_fbx_dst))
        if resolved_tex_in != final_tex_dst:
            shutil.copy(str(resolved_tex_in), str(final_tex_dst))

        print(f"Successfully saved {final_fbx_dst} ({final_fbx_dst.stat().st_size} bytes)")
        print(f"Saved companion {final_tex_dst} ({final_tex_dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

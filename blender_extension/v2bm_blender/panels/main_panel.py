"""3D Viewport → Sidebar (N Panel) → VideoIntoBlenderModel."""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.types import Context, Panel

from ..i18n import env_status_label, localize, stage_label, status_label, t
from ..operators.runtime import get_runtime
from ..preferences import get_preferences
from ..registration import safe_register_class, safe_unregister_class
from ..services.backend import tail_text


class V2BM_PT_main(Panel):
    bl_label = "VideoIntoBlenderModel"
    bl_idname = "V2BM_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VideoIntoBlenderModel"

    def draw(self, context: Context) -> None:
        layout = self.layout
        props = context.scene.v2bm
        prefs = get_preferences(context)
        runtime = get_runtime()
        busy = runtime is not None and not runtime.finished

        self._draw_setup(layout, prefs, props, busy)
        self._draw_input(layout, props, busy)
        self._draw_progress(layout, props, busy)

    def _draw_setup(self, layout, prefs, props, busy: bool) -> None:
        box = layout.box()
        box.label(text=t("section_setup"))
        box.prop(prefs, "ui_language", text=t("ui_language"))
        box.prop(prefs, "python_path", text=t("external_python"))
        box.prop(prefs, "project_dir", text=t("project_directory"))
        box.prop(props, "device", text=t("device"))
        row = box.row()
        row.enabled = not busy
        row.operator("v2bm.check_environment", text=t("check_environment"), icon="CHECKMARK")
        box.label(text=t("environment_fmt", status=env_status_label(props.env_status)))
        if props.env_message:
            box.label(text=localize(props.env_message))

    def _draw_input(self, layout, props, busy: bool) -> None:
        box = layout.box()
        box.label(text=t("section_input"))
        box.prop(props, "source_type", text=t("source_type"))
        box.prop(props, "source_path", text=t("source_file"))
        box.prop(props, "rig_path", text=t("humanoid_rig"))
        box.prop(props, "generate_preview", text=t("generate_preview"))
        row = box.row()
        row.enabled = not busy
        row.scale_y = 1.4
        row.operator("v2bm.run_pipeline", text=t("generate_motion"), icon="PLAY")

    def _draw_progress(self, layout, props, busy: bool) -> None:
        box = layout.box()
        box.label(text=t("section_progress"))
        status = props.job_status or "idle"
        box.label(text=t("status_fmt", status=status_label(status)))

        if busy or status in {"running", "cancelling"}:
            if props.job_stage:
                box.label(text=t("stage_fmt", stage=stage_label(props.job_stage)))
            try:
                box.progress(factor=float(props.job_progress), type="BAR")
            except Exception:
                box.prop(props, "job_progress", text=t("progress"), slider=True)
            log_path = Path(props.job_log_path) if props.job_log_path else None
            log = tail_text(log_path, max_lines=8) if log_path else ""
            if log:
                log_box = box.box()
                log_box.label(text=t("recent_log"))
                for line in log.splitlines()[-8:]:
                    log_box.label(text=line[:96])
            box.operator("v2bm.cancel_job", text=t("cancel"), icon="X")
            return

        if status in {"completed", "import_failed", "failed", "cancelled"}:
            if props.job_stage:
                box.label(text=t("stage_fmt", stage=stage_label(props.job_stage)))
            if status == "completed":
                box.label(text=t("generation_succeeded"))
            elif status == "import_failed":
                box.label(text=t("generation_import_failed"))
            elif status == "cancelled":
                box.label(text=t("cancelled"))
            if props.job_error:
                box.label(text=localize(props.job_error))
            if props.job_glb_path:
                box.label(text=t("glb_fmt", name=Path(props.job_glb_path).name))
            row = box.row()
            row.operator("v2bm.import_result", text=t("import_character"), icon="IMPORT")
            row = box.row()
            row.operator("v2bm.open_output", text=t("open_output"), icon="FILE_FOLDER")
            row.operator("v2bm.view_log", text=t("view_log"), icon="TEXT")
            return

        box.label(text=t("no_job_yet"))


def register() -> None:
    safe_register_class(V2BM_PT_main)


def unregister() -> None:
    safe_unregister_class(V2BM_PT_main)

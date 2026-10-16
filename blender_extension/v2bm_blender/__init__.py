"""VideoIntoBlenderModel Blender Extension: UI + external `v2bm` process control."""

from __future__ import annotations

bl_info = {
    "name": "VideoIntoBlenderModel",
    "author": "VideoIntoBlenderModel Contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > VideoIntoBlenderModel",
    "description": "Transfer human motion from video to Humanoid characters",
    "category": "Animation",
}

from . import operators, panels, preferences, properties


def register() -> None:
    try:
        unregister()
    except Exception:
        pass
    preferences.register()
    properties.register()
    operators.register()
    panels.register()


def unregister() -> None:
    try:
        panels.unregister()
    except Exception:
        pass
    try:
        operators.unregister()
    except Exception:
        pass
    try:
        properties.unregister()
    except Exception:
        pass
    try:
        preferences.unregister()
    except Exception:
        pass

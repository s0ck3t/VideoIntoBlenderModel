"""Automated 3-Way Side-by-Side Comparison Video and Keyframe Comparison Grid Generator.

Generates:
1. side_by_side_comparison.mp4:
   A 1920x696 3-panel comparison video with clean labelled headers:
   - Panel 1: Source Video (assets/video/urban_dancer_51314.mp4)
   - Panel 2: Baseline Render (outputs/20261014_184729_urban_dancer_51314/videos/blender_character_render.mp4)
   - Panel 3: Improved Render (outputs/20261014_improved_urban_dancer_51314/videos/blender_character_render.mp4)

2. keyframe_comparison_grid.png:
   A high-resolution 1920x3480 5-row keyframe comparison grid across the 3 panels,
   evaluating frames around 30, 90, 150, 170, and 216 (specifically highlighting the
   right forearm foreshortening resolution).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(font_size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load standard Windows sans-serif font with graceful fallback."""
    font_names = (
        ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf", "calibri.ttf"]
    )
    fonts_dir = Path("C:/Windows/Fonts")
    for name in font_names:
        p = fonts_dir / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), font_size)
            except Exception:
                pass
    return ImageFont.load_default()


def _crop_and_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop a video frame to 1:1 aspect ratio and resize to (target_w, target_h)."""
    h, w = frame.shape[:2]
    # Crop to square centered on horizontal axis
    if w > h:
        x0 = (w - h) // 2
        cropped = frame[:, x0 : x0 + h]
    elif h > w:
        y0 = (h - w) // 2
        cropped = frame[y0 : y0 + w, :]
    else:
        cropped = frame
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


def create_header_banner(
    width: int = 1920,
    height: int = 56,
    panel_titles: Sequence[str] = (
        "Source Video",
        "Baseline Render",
        "Improved Render",
    ),
    panel_subtitles: Sequence[str] = (
        "urban_dancer_51314.mp4 (240 frames)",
        "Baseline Pipeline (18:47:29)",
        "IK Optimiser + Two-Vector Hand",
    ),
    accent_colours: Sequence[tuple[int, int, int]] = (
        (0, 180, 216),    # Cyan
        (231, 111, 81),   # Coral
        (42, 157, 143),   # Teal
    ),
) -> np.ndarray:
    """Render a clean 3-panel header banner with titles, subtitles, and accent bars."""
    img = Image.new("RGB", (width, height), color=(20, 22, 26))
    draw = ImageDraw.Draw(img)

    panel_w = width // 3
    title_font = _load_font(18, bold=True)
    sub_font = _load_font(12, bold=False)

    for i in range(3):
        x0 = i * panel_w
        x1 = x0 + panel_w
        accent = accent_colours[i % len(accent_colours)]

        # Top accent strip (3px)
        draw.rectangle([x0, 0, x1, 3], fill=accent)

        # Title and subtitle centered in column
        title = panel_titles[i]
        sub = panel_subtitles[i]

        t_bbox = draw.textbbox((0, 0), title, font=title_font)
        s_bbox = draw.textbbox((0, 0), sub, font=sub_font)
        t_w = t_bbox[2] - t_bbox[0]
        s_w = s_bbox[2] - s_bbox[0]

        t_x = x0 + (panel_w - t_w) // 2
        s_x = x0 + (panel_w - s_w) // 2

        draw.text((t_x, 8), title, fill=(245, 247, 250), font=title_font)
        draw.text((s_x, 32), sub, fill=(160, 168, 180), font=sub_font)

        # Subtle vertical separator between panels
        if i > 0:
            draw.line([x0, 0, x0, height], fill=(45, 49, 57), width=1)

    # Bottom border line
    draw.line([0, height - 1, width, height - 1], fill=(45, 49, 57), width=1)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_comparison_video(
    source_video: Path,
    baseline_video: Path,
    improved_video: Path,
    out_video: Path,
    target_fps: float = 23.976,
) -> None:
    """Generate 1920x696 3-panel side-by-side comparison video."""
    print(f"Opening video inputs:\n  Source:   {source_video}\n  Baseline: {baseline_video}\n  Improved: {improved_video}")
    cap_src = cv2.VideoCapture(str(source_video))
    cap_base = cv2.VideoCapture(str(baseline_video))
    cap_imp = cv2.VideoCapture(str(improved_video))

    if not (cap_src.isOpened() and cap_base.isOpened() and cap_imp.isOpened()):
        raise RuntimeError("Failed to open one or more input video streams for comparison")

    n_frames_src = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT) or 240)
    n_frames_base = int(cap_base.get(cv2.CAP_PROP_FRAME_COUNT) or 240)
    n_frames_imp = int(cap_imp.get(cv2.CAP_PROP_FRAME_COUNT) or 240)
    total_frames = min(n_frames_src, n_frames_base, n_frames_imp)
    print(f"Total synchronized frames: {total_frames}")

    header_banner = create_header_banner(width=1920, height=56)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg pipe with -loglevel error to avoid pipe buffer deadlock
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", "1920x696",
        "-pix_fmt", "bgr24",
        "-r", f"{target_fps:.3f}",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(out_video.resolve()),
    ]
    pipe = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    badge_font = _load_font(13, bold=True)

    try:
        for frame_idx in range(total_frames):
            ret_s, f_src = cap_src.read()
            ret_b, f_base = cap_base.read()
            ret_i, f_imp = cap_imp.read()

            if not (ret_s and ret_b and ret_i):
                break

            panel_s = _crop_and_resize(f_src, 640, 640)
            panel_b = _crop_and_resize(f_base, 640, 640)
            panel_i = _crop_and_resize(f_imp, 640, 640)

            # Assemble 1920x640 content strip
            video_strip = np.hstack([panel_s, panel_b, panel_i])

            # Draw vertical separation lines
            cv2.line(video_strip, (640, 0), (640, 640), (45, 49, 57), 1)
            cv2.line(video_strip, (1280, 0), (1280, 640), (45, 49, 57), 1)

            # Stack header banner (56px) on top of video strip (640px) = 696px
            composite = np.vstack([header_banner, video_strip])

            # Add frame counter badge in bottom-right corner
            badge_text = f"Frame {frame_idx:03d} / {total_frames:03d}"
            # Render badge using PIL for clean antialiasing
            img_pil = Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            bx0 = 1920 - bw - 24
            by0 = 696 - bh - 16
            draw.rounded_rectangle([bx0 - 8, by0 - 4, bx0 + bw + 8, by0 + bh + 4], radius=4, fill=(16, 18, 22))
            draw.text((bx0, by0), badge_text, fill=(220, 226, 235), font=badge_font)

            frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            pipe.stdin.write(frame_bgr.tobytes())

        pipe.stdin.close()
        pipe.wait()
    finally:
        cap_src.release()
        cap_base.release()
        cap_imp.release()

    if out_video.exists() and out_video.stat().st_size > 0:
        print(f"Successfully generated comparison video: {out_video} ({out_video.stat().st_size:,} bytes)")
    else:
        stderr = pipe.stderr.read().decode("utf-8", errors="replace") if pipe.stderr else ""
        raise RuntimeError(f"Comparison video generation failed:\n{stderr}")


def render_keyframe_grid(
    source_video: Path,
    baseline_video: Path,
    improved_video: Path,
    out_grid_png: Path,
    keyframes: Sequence[int] = (30, 90, 150, 170, 216),
    keyframe_descriptions: Sequence[str] = (
        "Frame 30: Initial Pose & Arm Extension",
        "Frame 90: Dynamic Turn & Upper Body Heading",
        "Frame 150: Mid-Sequence Footwork & Leg Articulation",
        "Frame 170: Left Arm Extension & Forearm Pronation",
        "Frame 216: Right Forearm Foreshortening & Palm Orientation Resolution",
    ),
) -> None:
    """Generate high-resolution 1920x3480 5-row keyframe comparison grid."""
    print(f"Generating keyframe grid for frames {list(keyframes)}...")
    cap_src = cv2.VideoCapture(str(source_video))
    cap_base = cv2.VideoCapture(str(baseline_video))
    cap_imp = cv2.VideoCapture(str(improved_video))

    if not (cap_src.isOpened() and cap_base.isOpened() and cap_imp.isOpened()):
        raise RuntimeError("Failed to open video sources for keyframe grid generation")

    rows: list[np.ndarray] = []
    header_banner = create_header_banner(width=1920, height=56)
    row_title_font = _load_font(15, bold=True)

    try:
        for idx, (frame_num, desc) in enumerate(zip(keyframes, keyframe_descriptions)):
            cap_src.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            cap_base.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            cap_imp.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret_s, f_src = cap_src.read()
            ret_b, f_base = cap_base.read()
            ret_i, f_imp = cap_imp.read()

            if not (ret_s and ret_b and ret_i):
                raise RuntimeError(f"Could not read frame {frame_num} from inputs")

            panel_s = _crop_and_resize(f_src, 640, 640)
            panel_b = _crop_and_resize(f_base, 640, 640)
            panel_i = _crop_and_resize(f_imp, 640, 640)

            video_strip = np.hstack([panel_s, panel_b, panel_i])
            cv2.line(video_strip, (640, 0), (640, 640), (45, 49, 57), 1)
            cv2.line(video_strip, (1280, 0), (1280, 640), (45, 49, 57), 1)

            # Header for row: on the very first row, use the full column header;
            # on subsequent rows, include row description banner
            row_header = create_header_banner(
                width=1920,
                height=56,
                panel_titles=(
                    f"Source Video [F{frame_num}]",
                    f"Baseline Render [F{frame_num}]",
                    f"Improved Render [F{frame_num}]",
                ),
                panel_subtitles=(
                    desc,
                    "Baseline Pipeline",
                    "Sliding-Window IK Optimised",
                ),
            )
            row_composite = np.vstack([row_header, video_strip])

            # Overlay prominent frame badge on Improved panel
            img_pil = Image.fromarray(cv2.cvtColor(row_composite, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            badge_text = f"Frame {frame_num:03d} — {desc}"
            t_bbox = draw.textbbox((0, 0), badge_text, font=row_title_font)
            bw = t_bbox[2] - t_bbox[0]
            bh = t_bbox[3] - t_bbox[1]
            draw.rounded_rectangle([20, 696 - bh - 24, 20 + bw + 20, 696 - 12], radius=4, fill=(16, 18, 22))
            draw.text((30, 696 - bh - 20), badge_text, fill=(245, 247, 250), font=row_title_font)

            row_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            rows.append(row_bgr)
    finally:
        cap_src.release()
        cap_base.release()
        cap_imp.release()

    full_grid = np.vstack(rows)
    print(f"Grid composite shape: {full_grid.shape}")
    out_grid_png.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out_grid_png), full_grid)
    if not success or not out_grid_png.exists():
        raise RuntimeError(f"Failed to write keyframe grid to {out_grid_png}")
    print(f"Successfully generated keyframe grid: {out_grid_png} ({out_grid_png.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3-way side-by-side video and keyframe grid")
    parser.add_argument("--source", type=Path, default=Path("assets/video/urban_dancer_51314.mp4"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/20261014_184729_urban_dancer_51314/videos/blender_character_render.mp4"))
    parser.add_argument("--improved", type=Path, default=Path("outputs/20261014_improved_urban_dancer_51314/videos/blender_character_render.mp4"))
    parser.add_argument("--out-video", type=Path, default=Path("outputs/20261014_improved_urban_dancer_51314/videos/side_by_side_comparison.mp4"))
    parser.add_argument("--out-grid", type=Path, default=Path("outputs/20261014_improved_urban_dancer_51314/videos/keyframe_comparison_grid.png"))
    args = parser.parse_args()

    render_comparison_video(args.source, args.baseline, args.improved, args.out_video)
    render_keyframe_grid(args.source, args.baseline, args.improved, args.out_grid)


if __name__ == "__main__":
    main()

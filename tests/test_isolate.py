"""Unit and regression tests for v2bm.isolate.

Verifies multi-person tracking, identity persistence across swaps,
removal mask generation with target silhouette protection, and CLI options.
All code, comments, and docstrings strictly use British English conventions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from v2bm.isolate import (
    DetectedPose,
    MultiPersonTracker,
    build_removal_mask,
)
from v2bm.cli import build_parser


def _mock_pose(
    idx: int,
    cx: float,
    cy: float,
    height: float,
    colour: tuple[float, float, float] = (100.0, 150.0, 200.0),
    mask: np.ndarray | None = None,
) -> DetectedPose:
    """Construct a mock DetectedPose for deterministic tracker unit testing."""
    landmarks = np.zeros((33, 3), dtype=np.float64)
    bbox = (cx - 20.0, cy - height / 2.0, cx + 20.0, cy + height / 2.0)
    area = 40.0 * height
    return DetectedPose(
        index=idx,
        landmarks=landmarks,
        bbox=bbox,
        centroid=np.array([cx, cy], dtype=np.float64),
        height=height,
        area=area,
        torso_colour=np.array(colour, dtype=np.float64),
        segmentation_mask=mask,
    )


def test_tracker_target_selection_criteria() -> None:
    """Verify target selection under various criteria: left, right, tallest, shortest."""
    p_left = _mock_pose(0, cx=100.0, cy=200.0, height=300.0)
    p_right = _mock_pose(1, cx=400.0, cy=200.0, height=250.0)
    poses = [p_left, p_right]
    shape = (480, 640)

    # Left
    tracker_left = MultiPersonTracker(target_selector="left")
    matched_left = tracker_left.update(poses, shape)
    left_target = tracker_left.target_track_id
    assert matched_left[left_target].centroid[0] == pytest.approx(100.0)

    # Right
    tracker_right = MultiPersonTracker(target_selector="right")
    matched_right = tracker_right.update(poses, shape)
    right_target = tracker_right.target_track_id
    assert matched_right[right_target].centroid[0] == pytest.approx(400.0)

    # Tallest
    tracker_tallest = MultiPersonTracker(target_selector="tallest")
    matched_tallest = tracker_tallest.update(poses, shape)
    tallest_target = tracker_tallest.target_track_id
    assert matched_tallest[tallest_target].height == pytest.approx(300.0)

    # Shortest
    tracker_shortest = MultiPersonTracker(target_selector="shortest")
    matched_shortest = tracker_shortest.update(poses, shape)
    shortest_target = tracker_shortest.target_track_id
    assert matched_shortest[shortest_target].height == pytest.approx(250.0)


def test_tracker_identity_persistence_across_swaps() -> None:
    """Verify Hungarian tracking maintains identity when two performers swap lateral sides."""
    shape = (480, 640)
    colour_a = (150.0, 50.0, 50.0)  # maroon / dark red
    colour_b = (220.0, 200.0, 100.0)  # light cyan / blue

    # Frame 0: Dancer A on left (x=150), Dancer B on right (x=450)
    p_a0 = _mock_pose(0, cx=150.0, cy=250.0, height=320.0, colour=colour_a)
    p_b0 = _mock_pose(1, cx=450.0, cy=250.0, height=270.0, colour=colour_b)

    tracker = MultiPersonTracker(target_selector="left")
    matched0 = tracker.update([p_a0, p_b0], shape)
    target_id = tracker.target_track_id
    assert target_id is not None
    assert matched0[target_id].centroid[0] == pytest.approx(150.0)

    # Frame 1: Intermediate movement towards centre
    p_a1 = _mock_pose(0, cx=200.0, cy=250.0, height=318.0, colour=colour_a)
    p_b1 = _mock_pose(1, cx=400.0, cy=250.0, height=272.0, colour=colour_b)
    matched1 = tracker.update([p_a1, p_b1], shape)
    assert matched1[target_id].centroid[0] == pytest.approx(200.0)

    # Frame 2: Crossing centre
    p_a2 = _mock_pose(0, cx=290.0, cy=250.0, height=320.0, colour=colour_a)
    p_b2 = _mock_pose(1, cx=310.0, cy=250.0, height=270.0, colour=colour_b)
    matched2 = tracker.update([p_a2, p_b2], shape)
    assert matched2[target_id].centroid[0] == pytest.approx(290.0)

    # Frame 3: Completed lateral swap (Dancer A is now on the right at x=440, Dancer B on left at x=160)
    p_b3 = _mock_pose(0, cx=160.0, cy=250.0, height=270.0, colour=colour_b)
    p_a3 = _mock_pose(1, cx=440.0, cy=250.0, height=320.0, colour=colour_a)
    matched3 = tracker.update([p_b3, p_a3], shape)

    # Dancer A should still be matched to target_id despite being on the right and at index 1
    assert matched3[target_id].centroid[0] == pytest.approx(440.0)
    assert matched3[target_id].height == pytest.approx(320.0)


def test_build_removal_mask_silhouette_protection() -> None:
    """Verify build_removal_mask protects the target silhouette from inpainting encroachment."""
    h, w = 100, 100
    target_mask = np.zeros((h, w), dtype=np.float32)
    target_mask[30:70, 40:60] = 1.0
    p_target = _mock_pose(0, cx=50.0, cy=50.0, height=40.0, mask=target_mask)

    other_mask = np.zeros((h, w), dtype=np.float32)
    other_mask[30:70, 20:45] = 1.0
    p_other = _mock_pose(1, cx=32.0, cy=50.0, height=40.0, mask=other_mask)

    matched = {0: p_target, 1: p_other}
    removal_mask = build_removal_mask((h, w), matched, target_track_id=0, dilate_removal_px=5, protect_target_px=3)

    assert isinstance(removal_mask, np.ndarray)
    assert removal_mask.shape == (h, w)
    assert removal_mask.dtype == np.uint8

    # Non-target region clearly marked for inpainting
    assert np.any(removal_mask[:, 20:35] == 255)

    # Core target silhouette strictly protected (value must be 0)
    assert np.all(removal_mask[35:65, 42:58] == 0), "Target silhouette encroached by removal mask!"


def test_cli_parser_target_subject() -> None:
    """Verify CLI parser recognises --target-subject and isolate command."""
    parser = build_parser()

    # Verify run subparser with --target-subject
    args_run = parser.parse_args(["run", "--video", "dummy.mp4", "--target-subject", "left"])
    assert args_run.command == "run"
    assert args_run.target_subject == "left"

    # Verify isolate subparser
    args_iso = parser.parse_args(
        ["isolate", "--video", "in.mp4", "--output", "out.mp4", "--target-subject", "tallest"]
    )
    assert args_iso.command == "isolate"
    assert args_iso.video == "in.mp4"
    assert args_iso.output == "out.mp4"
    assert args_iso.target_subject == "tallest"

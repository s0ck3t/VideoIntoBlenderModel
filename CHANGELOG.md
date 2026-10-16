# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Blender 4.2+ Extension that launches `v2bm` in an external Python environment, streams progress, and imports the generated GLB.
- `v2bm run --output-dir`, `--progress-jsonl`, and `--no-preview` for add-on job control (existing CLI flags unchanged).

## [0.1.0] - 2026-09-04

### Added

- Retarget single-person motion videos to Humanoid-rigged characters.
- Recover and transfer a static 3D pose from a single-person image.
- Export reusable human skeleton motion as `skeleton_motion.npz`.
- Export per-bone Humanoid rotations as `rig_rotations.npz`.
- Export animated, skinned Humanoid characters as `.glb`.
- Generate human skeleton, Humanoid skeleton, character, and comparison previews.
- Reuse previously extracted skeleton motion with `--skeleton`.
- Validate dependencies and user-provided assets with `v2bm doctor`.
- Select CPU, CUDA, or Apple Silicon MPS for motion extraction.

[Unreleased]: https://localhost/VideoIntoBlenderModel/compare/v0.1.0...HEAD
[0.1.0]: https://localhost/VideoIntoBlenderModel/releases/tag/v0.1.0

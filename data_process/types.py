from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SampleIndices:
    t0: int
    preceding_indices: list[int]
    target_indices: list[int]
    candidate_indices: list[int]


EpisodeIndices = SampleIndices


@dataclass
class VideoGeometry:
    frames: Optional[list] = None
    depths: Optional[list] = None
    intrinsics: Optional[list] = None
    poses_c2w: Optional[list] = None
    masks: Optional[list] = None
    frame_indices: Optional[list] = None
    original_size: Optional[tuple[int, int]] = None
    processed_size: Optional[tuple[int, int]] = None

from __future__ import annotations

import random
from typing import Optional

from data_process.types import SampleIndices


def sample_frame_indices(
    num_frames: int,
    N_target: int,
    M_pre: int,
    min_gap_for_candidates: int = 0,
    rng: Optional[random.Random] = None,
) -> SampleIndices:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if N_target <= 0 or M_pre <= 0:
        raise ValueError("N_target and M_pre must be positive")
    if num_frames < N_target + M_pre:
        raise ValueError("Not enough frames to sample training sample")

    rng = rng or random.Random()
    t0_min = M_pre
    t0_max = num_frames - N_target
    if t0_min > t0_max:
        raise ValueError("Invalid sampling range for t0")

    t0 = rng.randint(t0_min, t0_max)
    target_indices = list(range(t0, t0 + N_target))
    preceding_indices = list(range(t0 - M_pre, t0))

    preceding_set = set(preceding_indices)
    target_set = set(target_indices)
    candidate_indices = [
        i for i in range(num_frames) if i not in preceding_set and i not in target_set
    ]

    if min_gap_for_candidates > 0:
        protected = sorted(preceding_indices + target_indices)

        def far_enough(idx: int) -> bool:
            return min(abs(idx - t) for t in protected) >= min_gap_for_candidates

        candidate_indices = [i for i in candidate_indices if far_enough(i)]

    return SampleIndices(
        t0=t0,
        preceding_indices=preceding_indices,
        target_indices=target_indices,
        candidate_indices=candidate_indices,
    )


sample_episode_indices = sample_frame_indices

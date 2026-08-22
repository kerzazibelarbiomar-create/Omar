from __future__ import annotations


def sample_clip(frames, *, num_samples=1):
    if not frames:
        return []
    if num_samples <= 1:
        return [frames[0]]
    return frames[:num_samples]

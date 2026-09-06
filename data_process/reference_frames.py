from __future__ import annotations

from typing import Iterable, Optional

from data_process.geometry import transform_points, unproject_depth_to_points, voxel_indices


def occupancy_from_frame(depth, K, c2w, voxel_size: float, valid_mask=None, dynamic_mask=None):
    if not isinstance(depth, (list, tuple)):
        depth = [[depth]]
    H = len(depth)
    if H and not isinstance(depth[0], (list, tuple)):
        depth = [[value] for value in depth]
        W = 1
    else:
        W = len(depth[0]) if H else 0
    mask = [[depth[h][w] > 0 for w in range(W)] for h in range(H)]
    if valid_mask is not None:
        mask = [[mask[h][w] and bool(valid_mask[h][w]) for w in range(W)] for h in range(H)]
    if dynamic_mask is not None:
        mask = [[mask[h][w] and not bool(dynamic_mask[h][w]) for w in range(W)] for h in range(H)]

    points_cam = unproject_depth_to_points(depth, K, mask=mask)
    if not points_cam:
        return set()

    points_world = transform_points(points_cam, c2w)
    vox = voxel_indices(points_world, voxel_size)
    return set(map(tuple, vox))


def iou_occupancy(a: set[tuple[int, int, int]], b: set[tuple[int, int, int]]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class RefSelectionResult:
    def __init__(self, indices: list[int], ious: list[float], stats: dict):
        self.indices = indices
        self.ious = ious
        self.stats = stats

    @property
    def count(self) -> int:
        return len(self.indices)

    def get_status_str(self) -> str:
        if self.count > 0:
            return f"ref={self.count}, best_iou={self.stats['best_iou']:.3f}"
        reason = self.stats.get("no_ref_reason", "unknown")
        best = self.stats.get("best_iou", 0)
        thresh = self.stats.get("threshold", 0)
        if reason == "max_refs_zero":
            return "ref=0 (max_refs=0)"
        if reason == "no_candidates":
            return "ref=0 (no candidates)"
        if reason == "iou_below_threshold":
            return f"ref=0 (best_iou={best:.3f}<{thresh:.3f})"
        return f"ref=0 ({reason})"


def select_reference_frames(
    candidate_indices: Iterable[int],
    target_indices: Iterable[int],
    depths,
    intrinsics,
    poses_c2w,
    voxel_size: float,
    stride: int,
    iou_threshold: float,
    max_refs: int,
    valid_masks=None,
    dynamic_masks=None,
    return_result: bool = False,
):
    stats = {
        "threshold": iou_threshold,
        "max_refs": max_refs,
        "stride": stride,
        "voxel_size": voxel_size,
    }

    if max_refs <= 0:
        stats["no_ref_reason"] = "max_refs_zero"
        stats["best_iou"] = 0.0
        if return_result:
            return RefSelectionResult([], [], stats)
        return [], []

    target_list = list(target_indices)
    candidates = list(candidate_indices)
    if stride > 1:
        candidates = candidates[::stride]

    stats["num_targets"] = len(target_list)
    stats["num_candidates"] = len(candidates)

    if not candidates:
        stats["no_ref_reason"] = "no_candidates"
        stats["best_iou"] = 0.0
        if return_result:
            return RefSelectionResult([], [], stats)
        return [], []

    target_occs = [
        occupancy_from_frame(
            depth=depths[idx],
            K=intrinsics[idx],
            c2w=poses_c2w[idx],
            voxel_size=voxel_size,
            valid_mask=None if valid_masks is None else valid_masks[idx],
            dynamic_mask=None if dynamic_masks is None else dynamic_masks[idx],
        )
        for idx in target_list
    ]

    candidate_occs = {}
    for idx in candidates:
        candidate_occs[idx] = occupancy_from_frame(
            depth=depths[idx],
            K=intrinsics[idx],
            c2w=poses_c2w[idx],
            voxel_size=voxel_size,
            valid_mask=None if valid_masks is None else valid_masks[idx],
            dynamic_mask=None if dynamic_masks is None else dynamic_masks[idx],
        )

    scored = []
    all_ious = []
    for c_idx in candidates:
        c_occ = candidate_occs[c_idx]
        max_iou = 0.0
        for t_occ in target_occs:
            iou = iou_occupancy(c_occ, t_occ)
            if iou > max_iou:
                max_iou = iou
        all_ious.append((c_idx, max_iou, len(c_occ)))
        if max_iou >= iou_threshold:
            scored.append((c_idx, max_iou))

    best_iou = max(x[1] for x in all_ious) if all_ious else 0.0
    avg_iou = sum(x[1] for x in all_ious) / len(all_ious) if all_ious else 0.0
    stats["best_iou"] = best_iou
    stats["avg_iou"] = avg_iou

    if len(scored) == 0:
        stats["no_ref_reason"] = "iou_below_threshold"

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:max_refs]
    indices = [s[0] for s in selected]
    ious = [s[1] for s in selected]

    if return_result:
        return RefSelectionResult(indices, ious, stats)
    return indices, ious

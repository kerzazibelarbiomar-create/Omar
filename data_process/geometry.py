from __future__ import annotations

from typing import Optional


def _shape(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        if isinstance(value[0], (list, tuple)):
            return (len(value), len(value[0]))
        return (len(value),)
    return ()


def _to_matrix(value):
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def unproject_depth_to_points(depth, K, mask=None, return_pixels=False):
    if _shape(depth) != (len(depth), len(depth[0])):
        raise ValueError("depth must be HxW")
    if _shape(K) != (3, 3):
        raise ValueError("K must be 3x3")

    H, W = len(depth), len(depth[0])
    if mask is None:
        mask = [[depth[h][w] > 0 for w in range(W)] for h in range(H)]
    else:
        mask = [[bool(mask[h][w]) and depth[h][w] > 0 for w in range(W)] for h in range(H)]

    points = []
    pixels = []
    for h in range(H):
        for w in range(W):
            if not mask[h][w]:
                continue
            z = float(depth[h][w])
            fx = float(K[0][0])
            fy = float(K[1][1])
            cx = float(K[0][2])
            cy = float(K[1][2])
            x = (w - cx) * z / fx
            y = (h - cy) * z / fy
            points.append([x, y, z])
            pixels.append([w, h])

    if return_pixels:
        return points, pixels
    return points


def transform_points(points, transform):
    if not points:
        return []
    transform = _to_matrix(transform)
    if _shape(transform) != (4, 4):
        raise ValueError("transform must be 4x4")

    result = []
    for point in points:
        x, y, z = point[:3]
        px = transform[0][0] * x + transform[0][1] * y + transform[0][2] * z + transform[0][3]
        py = transform[1][0] * x + transform[1][1] * y + transform[1][2] * z + transform[1][3]
        pz = transform[2][0] * x + transform[2][1] * y + transform[2][2] * z + transform[2][3]
        result.append([px, py, pz])
    return result


def project_points(points, K):
    if not points:
        return [], []
    if _shape(K) != (3, 3):
        raise ValueError("K must be 3x3")

    uv = []
    z_values = []
    for point in points:
        x, y, z = point[:3]
        fx = float(K[0][0])
        fy = float(K[1][1])
        cx = float(K[0][2])
        cy = float(K[1][2])
        if z > 0:
            u = (x / z) * fx + cx
            v = (y / z) * fy + cy
        else:
            u = float("nan")
            v = float("nan")
        uv.append([u, v])
        z_values.append(z)
    return uv, z_values


def voxel_downsample(points, voxel_size: float):
    if voxel_size <= 0:
        return points
    if not points:
        return []

    seen = {}
    for point in points:
        voxel = tuple(int(p // voxel_size) for p in point)
        seen.setdefault(voxel, point)
    return list(seen.values())


def voxel_indices(points, voxel_size: float):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if not points:
        return []
    return [tuple(int(p // voxel_size) for p in point) for point in points]

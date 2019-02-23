import math
from typing import Dict, List

import numpy as np

from constants import VERT_SLANGE_ANGLE, PATTERN_SIZE


def filter_duplicates(points: np.ndarray):
    """
    :param points: XY sample trial
    :return: points with no subsequent duplicates
    """
    duplicates = (points[1:] == points[:-1]).all(axis=1)
    duplicates = np.where(duplicates)[0]
    points = np.delete(points, duplicates, axis=0)
    return points


def make_patterns_fixed_size(patterns: Dict[str, List[np.ndarray]]):
    """
    :param patterns: dict of sample trials to be used as patterns
    :return: patterns, each of which has PATTERN_SIZE XY pairs of points
    """
    patterns_fixed_size = {}
    for word in patterns.keys():
        patterns_fixed_size[word] = []
        for trial in patterns[word]:
            indices = np.linspace(0, len(trial) - 1, num=PATTERN_SIZE, endpoint=True, dtype=np.int32)
            trial = trial[indices]
            patterns_fixed_size[word].append(trial)
    return patterns_fixed_size


def correct_slant(points: np.ndarray):
    """
    Inplace slant correction.
    :param points: XY sample trial
    """
    points = np.asarray(points, dtype=np.float32)
    vert_slant_radians = math.radians(VERT_SLANGE_ANGLE)
    vert_slant_cotang = 1 / math.tan(vert_slant_radians)
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    take = np.abs(dy / dx) > vert_slant_cotang
    dx = dx[take]
    dy = dy[take]
    dx[dy < 0] *= -1
    dy[dy < 0] *= -1

    slant_dx, slant_dy = np.c_[dx, dy].sum(axis=0)
    if slant_dy == 0:
        return
    shear = -slant_dx / slant_dy
    points[:, 0] = points[:, 0] + points[:, 1] * shear


def is_inside_unit_box(points: np.ndarray, eps=1e-6) -> bool:
    return np.logical_and(points >= -eps, points <= 1. + eps).all()


def normalize(points: np.ndarray, keep_aspect_ratio=True):
    """
    Inplace centering inside a unit box.
    :param points: XY sample trial
    :param keep_aspect_ratio: keep aspect ratio unchanged
    """
    points = np.asarray(points, dtype=np.float32)
    x, y = points.T
    ymin, xmin, ymax, xmax = y.min(), x.min(), y.max(), x.max()
    scale_x = 1 / (xmax - xmin)
    scale_y = 1 / (ymax - ymin)
    if keep_aspect_ratio:
        scale_x = scale_y = min(scale_x, scale_y)
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    points[:, 0] = 0.5 + (x - xc) * scale_x
    points[:, 1] = 0.5 + (y - yc) * scale_y
    assert is_inside_unit_box(points)

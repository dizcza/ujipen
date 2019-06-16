import math
from typing import Dict, List

import numpy as np

from constants import VERT_SLANT_ANGLE, PATTERN_SIZE


def filter_duplicates(points: List[np.ndarray]):
    """
    :param points: list of strokes of XY points
    :return: points with no subsequent duplicates
    """
    points_filtered = []
    for stroke_points in points:
        duplicates = (stroke_points[1:] == stroke_points[:-1]).all(axis=1)
        duplicates = np.where(duplicates)[0]
        stroke_points = np.delete(stroke_points, duplicates, axis=0)
        points_filtered.append(stroke_points)
    return points_filtered


def get_fixed_stroke_numpoints(stroke_numpoints, total_points: int):
    stroke_numpoints = np.floor(stroke_numpoints).astype(int)
    stroke_numpoints = np.maximum(stroke_numpoints, 1)  # at least 1 point in a stroke
    stroke_numpoints[-1] = total_points - sum(stroke_numpoints[:-1])
    while stroke_numpoints[-1] <= 0:
        stroke_numpoints[stroke_numpoints.argmax()] -= 1
        stroke_numpoints[-1] += 1
    assert np.all(stroke_numpoints > 0) and np.sum(stroke_numpoints) == total_points
    return stroke_numpoints


def make_patterns_fixed_size(patterns: Dict[str, List[np.ndarray]], total_points=PATTERN_SIZE):
    """
    Dummy function. Use equally_spaced_points instead.
    :param patterns: dict of sample trials to be used as patterns
    :param total_points: num of points
    :return: patterns, each of which has `total_points` XY pairs of points
    """
    patterns_fixed_size = {}
    for word in patterns.keys():
        patterns_fixed_size[word] = []
        for trial in patterns[word]:
            stroke_numpoints = np.fromiter(map(len, trial), dtype=int)
            stroke_numpoints = total_points * stroke_numpoints.astype(np.float32) / stroke_numpoints.sum()
            stroke_numpoints = get_fixed_stroke_numpoints(stroke_numpoints, total_points=total_points)
            trial_fixed_size = []
            for stroke_points, stroke_numpoint in zip(trial, stroke_numpoints):
                indices = np.linspace(0, len(stroke_points) - 1, num=stroke_numpoint, endpoint=True, dtype=np.int32)
                trial_fixed_size.append(stroke_points[indices])
            assert sum(map(len, trial_fixed_size)) == total_points
            patterns_fixed_size[word].append(trial_fixed_size)
    return patterns_fixed_size


def equally_spaced_points(points: List[np.ndarray], total_points: int):
    """
    :param points: list of strokes of XY points
    :param total_points: num of points
    :return: points with the total of `total_points` XY pairs of points,
             distributed equally along the path that connects segments.
    """
    if sum(map(len, points)) == total_points:
        return points
    segment_vectors_all = [np.diff(stroke_points, axis=0) for stroke_points in points]
    segment_lengths_all = [np.linalg.norm(segment_vectors, axis=1) for segment_vectors in segment_vectors_all]
    stroke_lengths = np.fromiter(map(sum, segment_lengths_all), dtype=np.float32)
    stroke_numpoints = total_points * stroke_lengths / stroke_lengths.sum()
    stroke_numpoints = get_fixed_stroke_numpoints(stroke_numpoints, total_points=total_points)

    points_equally_spaced = []
    for stroke_id, stroke_points in enumerate(points):
        segment_vectors = segment_vectors_all[stroke_id]
        segment_lengths = segment_lengths_all[stroke_id]
        segment_cumsum = np.r_[0, segment_lengths.cumsum()]
        segment_coords = np.linspace(0, stroke_lengths[stroke_id], num=stroke_numpoints[stroke_id] - 1, endpoint=False)
        stroke_equally_spaced = []
        for coord in segment_coords:
            segment_bin = np.digitize(coord, bins=segment_cumsum, right=False) - 1
            segment_part = (coord - segment_cumsum[segment_bin]) / segment_lengths[segment_bin]
            assert 0 <= segment_part < 1
            vec_add = segment_vectors[segment_bin] * segment_part
            point_new = stroke_points[segment_bin] + vec_add
            stroke_equally_spaced.append(point_new)
        stroke_equally_spaced.append(stroke_points[-1])
        stroke_equally_spaced = np.array(stroke_equally_spaced)
        points_equally_spaced.append(stroke_equally_spaced)
    assert sum(map(len, points_equally_spaced)) == total_points
    return points_equally_spaced


def equally_spaced_points_patterns(patterns: Dict[str, List[List[np.ndarray]]], total_points=PATTERN_SIZE):
    """
    :param patterns: dict of sample trials to be used as patterns
    :param total_points: num of points
    :return: patterns, each of which has `total_points` XY pairs of points,
             distributed equally along the path that connects segments.
    """
    patterns_fixed_size = {}
    for word in patterns.keys():
        patterns_fixed_size[word] = []
        for trial in patterns[word]:
            trial = equally_spaced_points(trial, total_points=total_points)
            patterns_fixed_size[word].append(trial)
    return patterns_fixed_size


def correct_slant(points: List[np.ndarray]):
    """
    Inplace slant correction.
    :param points: list of strokes of XY points
    """
    vert_slant_radians = math.radians(VERT_SLANT_ANGLE)
    vert_slant_cotang = 1 / math.tan(vert_slant_radians)
    segment_vectors = [np.diff(stroke_points, axis=0) for stroke_points in points]
    segment_vectors = np.vstack(segment_vectors)
    dx, dy = segment_vectors.T
    take = np.abs(dy / dx) > vert_slant_cotang
    dx = dx[take]
    dy = dy[take]
    dx[dy < 0] *= -1
    dy[dy < 0] *= -1

    slant_dx, slant_dy = np.c_[dx, dy].sum(axis=0)
    if slant_dy == 0:
        return
    shear = -slant_dx / slant_dy
    for stroke_points in points:
        stroke_points[:, 0] = stroke_points[:, 0] + stroke_points[:, 1] * shear


def is_inside_box(points: List[np.ndarray], box, eps=1e-6) -> bool:
    points = np.vstack(points)
    box = np.asarray(box, dtype=np.float32)
    return np.logical_and(box[0] - eps <= points, points <= box[1] + eps).all()


def normalize(points: List[np.ndarray], box=((-1, -1), (1, 1)), keep_aspect_ratio=True):
    """
    Inplace centering inside a box.
    :param points: list of strokes of XY points
    :param box: normalize points to this bounding box (top left, bottom right)
    :param keep_aspect_ratio: keep aspect ratio unchanged
    """
    points_merged = np.vstack(points)
    x, y = points_merged.T
    ymin, xmin, ymax, xmax = y.min(), x.min(), y.max(), x.max()
    box = np.asarray(box, dtype=np.float32)
    box_width, box_height = box[1] - box[0]
    box_center = np.mean(box, axis=0)
    scale_x = box_width / (xmax - xmin)
    scale_y = box_height / (ymax - ymin)
    if keep_aspect_ratio:
        scale_x = scale_y = min(scale_x, scale_y)
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    for stroke_points in points:
        stroke_points[:, 0] = box_center[0] + (stroke_points[:, 0] - xc) * scale_x
        stroke_points[:, 1] = box_center[1] + (stroke_points[:, 1] - yc) * scale_y
    assert is_inside_box(points, box=box)

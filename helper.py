import string
from typing import Iterable, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from constants import PATTERN_SIZE


def drop_items(items: list, drop_ids: Iterable[int]):
    return [items[i] for i in range(len(items)) if i not in drop_ids]


def draw_sample(sample: List[np.ndarray]):
    colors = ['blue', 'orange', 'cyan', 'black', 'magenta']
    margin = 0.02
    sample_stacked = np.vstack(sample)
    x_min, y_min = sample_stacked.min(axis=0)
    x_max, y_max = sample_stacked.max(axis=0)

    for stroke_id, stroke_points in enumerate(sample):
        x, y = stroke_points.T
        y = y_max - y
        color = colors[stroke_id % len(colors)]
        plt.plot(x, y, color=color)
        plt.scatter(x, y, color=color, s=3)
    plt.xlim(left=x_min - margin, right=x_max + margin)
    plt.ylim(bottom=0 - margin, top=y_max - y_min + margin)
    plt.axis('off')


def create_edge_rectangles_patch(sample: List[np.ndarray], rect_size=(0.03, 0.03)):
    rect_size = np.asarray(rect_size, dtype=np.float32)
    rect_colors = []
    rects = []
    y_max = np.vstack(sample)[:, 1].max()
    for stroke_points in sample:
        stroke_points = stroke_points.copy()
        stroke_points[:, 1] = y_max - stroke_points[:, 1]
        for pid in (0, -1):
            rect_edge = mpatches.Rectangle(stroke_points[pid] - rect_size / 2, *rect_size)
            rects.append(rect_edge)
        rect_colors.extend(['g', 'r'])
    pc = PatchCollection(rects, facecolors=rect_colors)
    return pc


def check_unique_patterns(patterns: dict, n_points=PATTERN_SIZE) -> bool:
    verified = ''.join(sorted(patterns.keys())) == string.ascii_lowercase
    for word in patterns.keys():
        for trial in patterns[word]:
            verified &= sum(map(len, trial)) == n_points
    return verified

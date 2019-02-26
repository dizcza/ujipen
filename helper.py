import math
from typing import Iterable, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection


def take_matrix_by_mask(dist_matrix: np.ndarray, mask_take):
    positions = np.where(mask_take)[0]
    dist_matrix = dist_matrix.copy()
    for axis in (0, 1):
        dist_matrix = np.take(dist_matrix, indices=positions, axis=axis)
    return dist_matrix


def take_trials_by_mask(trials: list, mask_take):
    return [_trial for _trial, leave_element in zip(trials, mask_take) if leave_element]


def drop_items(items: list, drop_ids: Iterable[int]):
    return [items[i] for i in range(len(items)) if i not in drop_ids]


def draw_sample(sample: List[np.ndarray]):
    colors = ['blue', 'orange', 'cyan', 'black', 'magenta']
    margin = 0.02
    y_max = np.vstack(sample)[:, 1].max()
    for stroke_id, stroke_points in enumerate(sample):
        x, y = stroke_points.T
        y = y_max - y
        color = colors[stroke_id % len(colors)]
        plt.plot(x, y, color=color)
        plt.scatter(x, y, color=color, s=3)
    plt.xlim(left=0 - margin, right=1 + margin)
    plt.ylim(bottom=0 - margin, top=1 + margin)
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


def display(word_points, labels=None, dist_matrix=None):
    if labels is None:
        labels = np.ones(len(word_points), dtype=np.int32)
    rect_size_init = np.array([0.03, 0.03])
    for label in np.unique(labels):
        plt.figure(label)
        cluster_points = []
        for pid in range(len(word_points)):
            if labels[pid] == label:
                cluster_points.append(word_points[pid])
        rows = math.floor(math.sqrt(len(cluster_points)))
        cols = math.ceil(len(cluster_points) / rows)
        rect_size = rect_size_init * rows

        if dist_matrix is not None:
            dist_matrix_cluster = take_matrix_by_mask(dist_matrix, mask_take=labels == label)
            min_inter_dist_id = dist_matrix_cluster.sum(axis=0).argmin()
        else:
            min_inter_dist_id = -1

        for i, sample in enumerate(cluster_points):
            ax = plt.subplot(rows, cols, i + 1)
            draw_sample(sample)
            rects = [mpatches.Rectangle(sample[pid] - rect_size / 2, *rect_size) for pid in (0, -1)]
            pc = PatchCollection(rects, facecolors=['g', 'r'])
            ax.add_collection(pc)
            if i == min_inter_dist_id:
                rect = mpatches.Rectangle((0, 0), width=1, height=1, fill=False, fc='none', ec='black', lw=2)
                ax.add_patch(rect)
        plt.suptitle(f'Cluster {label}')
    plt.show()

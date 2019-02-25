import math

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


def draw_sample(sample: np.ndarray):
    margin = 0.02
    x, y = sample.T
    y = y.max() - y
    plt.plot(x, y)
    plt.xlim(left=0 - margin, right=1 + margin)
    plt.ylim(bottom=0 - margin, top=1 + margin)
    plt.axis('off')


def display(word_points, labels=None, dist_matrix=None, between_stroke_thr=0.2):
    if labels is None:
        labels = np.ones(len(word_points), dtype=np.int32)
    margin = 0.02
    colors = ['blue', 'orange', 'cyan', 'black', 'magenta']
    rect_size_init = np.array([0.03, 0.03])
    for label in np.unique(labels):
        plt.figure()
        cluster_points = [word_points[i] for i in range(len(word_points)) if labels[i] == label]
        rows = math.floor(math.sqrt(len(cluster_points)))
        cols = math.ceil(len(cluster_points) / rows)
        rect_size = rect_size_init * rows

        if dist_matrix is not None:
            dist_matrix_cluster = take_matrix_by_mask(dist_matrix, mask_take=labels == label)
            min_inter_dist_id = dist_matrix_cluster.sum(axis=0).argmin()
        else:
            min_inter_dist_id = -1

        for i, sample in enumerate(cluster_points):
            sample = sample.copy()
            ax = plt.subplot(rows, cols, i + 1)
            dv = sample[1:] - sample[:-1]
            dist = np.linalg.norm(dv, axis=1)
            corners = np.where(dist > between_stroke_thr)[0]
            corners += 1
            sample[:, 1] = sample[:, 1].max() - sample[:, 1]
            for chunk_id, chunk in enumerate(np.split(sample, corners)):
                x, y = chunk.T
                plt.plot(x, y, color=colors[chunk_id % len(colors)])
            rects = [mpatches.Rectangle(sample[pid] - rect_size / 2, *rect_size) for pid in (0, -1)]
            pc = PatchCollection(rects, facecolors=['g', 'r'])
            ax.add_collection(pc)
            if i == min_inter_dist_id:
                rect = mpatches.Rectangle((0, 0), width=1, height=1, fill=False, fc='none', ec='black', lw=2)
                ax.add_patch(rect)

            plt.xlim(left=0 - margin, right=1 + margin)
            plt.ylim(bottom=0 - margin, top=1 + margin)
            plt.axis('off')
        plt.suptitle(f'Label {label}')
    plt.show()

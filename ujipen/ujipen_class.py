import math
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from helper import take_matrix_by_mask, take_trials_by_mask, draw_sample, create_edge_rectangles_patch
from ujipen.loader import ujipen_read, _save_ujipen, filter_alphabet, ujipen_correct_slant, ujipen_normalize, \
    save_intra_dist, check_shapes, ujipen_drop_from_dropped_list, ujipen_equally_spaced_points
from ujipen.ujipen_constants import *
from ujipen.ujipen_constants import UJIPEN_DROPPED_LIST


def onclick_drop_callback(event):
    if event.inaxes is None:
        return
    label = event.canvas.figure.number
    session_drop = event.inaxes.get_title()
    dropped_list = []
    if UJIPEN_DROPPED_LIST.exists():
        with open(UJIPEN_DROPPED_LIST) as f:
            dropped_list = f.readlines()
    dropped_list = set(dropped_list)
    dropped_list.add(session_drop + '\n')
    dropped_list = sorted(dropped_list)
    with open(UJIPEN_DROPPED_LIST, 'w') as f:
        f.writelines(dropped_list)
    print(f"Added '{session_drop}' label={label} in a dropped list.")


class UJIPen:

    def __init__(self, force_read=False, equally_spaced=False):
        if force_read:
            data = ujipen_read()
            filter_alphabet(data)
            ujipen_correct_slant(data)
            ujipen_normalize(data)
            if equally_spaced:
                ujipen_equally_spaced_points(data)
            save_intra_dist(data)
            _save_ujipen(data, path=UJIPEN_PKL)
        with open(UJIPEN_PKL, 'rb') as f:
            self.data = pickle.load(f)
        ujipen_drop_from_dropped_list(self.data)
        check_shapes(self.data)

    @property
    def num_patterns(self):
        patterns = self.get_min_intra_dist_patterns()
        total_patterns = sum(map(len, patterns.values()))
        return total_patterns

    def display_clustering(self, drop_onclick=False):
        for word in self.data["train"].keys():
            self.display(word, drop_onclick=drop_onclick)

    def get_min_intra_dist_patterns(self):
        patterns = {}
        for word in self.data["train"].keys():
            patterns[word] = []
            labels = self.data["train"][word].get(LABELS_KEY, None)
            if labels is None:
                continue
            dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
            for label_unique in set(labels):
                mask_cluster = labels == label_unique
                dist_matrix_cluster = take_matrix_by_mask(dist_matrix, mask_cluster)
                points_cluster = take_trials_by_mask(self.data["train"][word][TRIALS_KEY], mask_cluster)
                best_sample_id = dist_matrix_cluster.sum(axis=0).argmin()
                patterns[word].append(points_cluster[best_sample_id])
        return patterns

    def get_samples(self, fold="train"):
        return {
            word: self.data[fold][word][TRIALS_KEY] for word in self.data[fold].keys()
        }

    def display(self, word, labels=None, drop_onclick=False):
        if labels is None:
            labels = self.data["train"][word].get(LABELS_KEY, None)
        word_points = self.data["train"][word][TRIALS_KEY]
        dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
        sample_ids = self.data["train"][word][SESSION_KEY]
        if labels is None:
            labels = np.ones(len(word_points), dtype=np.int32)
        rect_size_init = np.array([0.03, 0.03])
        for label in np.unique(labels):
            fig = plt.figure(label)
            if drop_onclick:
                fig.canvas.mpl_connect('button_press_event', onclick_drop_callback)
            cluster_points = []
            cluster_sample_ids = []
            for pid in range(len(word_points)):
                if labels[pid] == label:
                    cluster_points.append(word_points[pid])
                    cluster_sample_ids.append(sample_ids[pid])
            rows = math.floor(math.sqrt(len(cluster_points)))
            cols = math.ceil(len(cluster_points) / rows)
            rect_size = rect_size_init * rows

            dist_matrix_cluster = take_matrix_by_mask(dist_matrix, mask_take=labels == label)
            min_inter_dist_id = dist_matrix_cluster.sum(axis=0).argmin()

            for i, sample in enumerate(cluster_points):
                ax = plt.subplot(rows, cols, i + 1)
                draw_sample(sample)
                ax.add_collection(create_edge_rectangles_patch(sample, rect_size=rect_size))
                if i == min_inter_dist_id:
                    rect = mpatches.Rectangle((0, 0), width=1, height=1, fill=False, fc='none', ec='black', lw=2)
                    ax.add_patch(rect)
                plt.title(cluster_sample_ids[i], fontsize=5)
            plt.suptitle(f'Label {label}')
        plt.show()

    def show_trial_size(self, patterns_only=False):
        sizes = []
        if patterns_only:
            samples = self.get_min_intra_dist_patterns()
        else:
            samples = self.get_samples()
        for word, trials in samples.items():
            sizes.extend(map(len, trials))
        plt.hist(sizes)
        plt.show()


if __name__ == '__main__':
    ujipen = UJIPen()
    print(f"UJIPen num. of patterns: {ujipen.num_patterns}")
    # ujipen.show_trial_size()
    ujipen.display_clustering(drop_onclick=True)

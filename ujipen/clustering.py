import math
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from ujipen.constants import *
from ujipen.loader import read_ujipen, _save_ujipen, filter_alphabet, correct_slant, normalize, save_intra_dist, \
    check_shapes


def display(points, labels, margin=0.02):
    colors = ['blue', 'orange', 'cyan', 'black', 'magenta']
    rect_size_init = np.array([0.03, 0.03])
    for label in np.unique(labels):
        plt.figure()
        label_points = [points[i] for i in range(len(points)) if labels[i] == label]
        rows = math.floor(math.sqrt(len(label_points)))
        cols = math.ceil(len(label_points) / rows)
        rect_size = rect_size_init * rows
        for i, sample in enumerate(label_points, start=1):
            sample = sample.copy()
            ax = plt.subplot(rows, cols, i)
            dv = sample[1:] - sample[:-1]
            dist = np.linalg.norm(dv, axis=1)
            corners = np.where(dist > 0.2)[0]
            corners += 1
            sample[:, 1] = sample[:, 1].max() - sample[:, 1]
            for chunk_id, chunk in enumerate(np.split(sample, corners)):
                x, y = chunk.T
                plt.plot(x, y, color=colors[chunk_id])
            rects = [mpatches.Rectangle(sample[pid] - rect_size / 2, *rect_size) for pid in (0, -1)]
            pc = PatchCollection(rects, facecolors=['g', 'r'])
            ax.add_collection(pc)
            plt.xlim(left=0 - margin, right=1 + margin)
            plt.ylim(bottom=0 - margin, top=1 + margin)
            plt.axis('off')
        plt.suptitle(f'Label {label}')
    plt.show()


class UJIPen:

    def __init__(self, force_read=False):
        if force_read:
            data = read_ujipen()
            filter_alphabet(data)
            correct_slant(data)
            normalize(data)
            save_intra_dist(data)
            _save_ujipen(data, path=UJIPEN_PKL)
        with open(UJIPEN_PKL, 'rb') as f:
            self.data = pickle.load(f)
        check_shapes(self.data)

    def drop_labels(self, word: str, labels_drop):
        word_points = self.data["train"][word][TRIALS_KEY]
        labels = self.data["train"][word][LABELS_KEY]
        dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
        assert len(word_points) == len(labels) == dist_matrix.shape[0] == dist_matrix.shape[1]
        indices_drop = np.where(np.isin(labels, labels_drop))[0]
        if len(indices_drop) == 0:
            return
        word_points = [word_points[i] for i in range(len(word_points)) if labels[i] not in labels_drop]
        labels = np.delete(labels, indices_drop)
        for axis in (0, 1):
            dist_matrix = np.delete(dist_matrix, indices_drop, axis=axis)
        assert len(word_points) == len(labels) == dist_matrix.shape[0] == dist_matrix.shape[1]
        self.data["train"][word][TRIALS_KEY] = word_points
        self.data["train"][word][LABELS_KEY] = labels
        self.data["train"][word][INTRA_DIST_KEY] = dist_matrix
        _save_ujipen(self.data)
        print(f"Word {word}: dropped {labels_drop}")

    @staticmethod
    def take_matrix_by_mask(dist_matrix: np.ndarray, mask_take):
        positions = np.where(mask_take)[0]
        dist_matrix = dist_matrix.copy()
        for axis in (0, 1):
            dist_matrix = np.take(dist_matrix, indices=positions, axis=axis)
        return dist_matrix

    @staticmethod
    def take_trials_by_mask(trials: list, mask_take):
        return [_trial for _trial, leave_element in zip(trials, mask_take) if leave_element]

    def compute_clustering_factor(self, labels, dist_matrix):
        assert len(labels) == dist_matrix.shape[0] == dist_matrix.shape[1]
        intra_dist = 0
        inter_dist = 0
        for label_unique in set(labels):
            mask = labels == label_unique
            if mask.sum() == 1:
                # skip clusters with a single sample
                continue
            intra_matrix = self.take_matrix_by_mask(dist_matrix, mask)
            inter_matrix = self.take_matrix_by_mask(dist_matrix, ~mask)
            intra_dist += intra_matrix.sum()
            inter_dist += inter_matrix.sum()
        factor = intra_dist / inter_dist
        return factor

    def cluster(self, max_clusters=10, intra_inter_thr_ratio=0.1, visualize=False):
        assert max_clusters >= 2, "At least 2 clusters per word"
        for word in self.data["train"].keys():
            dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
            clustering_factor = np.repeat(np.inf, repeats=max_clusters + 1)
            mask_non_single_experiment = {}
            labels_experiment = {}
            for n_clusters_cand in range(2, max_clusters + 1):
                predictor = AgglomerativeClustering(n_clusters=n_clusters_cand, affinity='precomputed',
                                                    linkage='average')
                labels = predictor.fit_predict(dist_matrix)
                labels_unique, counts = np.unique(labels, return_counts=True)
                labels_unique = labels_unique[counts > 1]  # drop outliers
                mask_non_single = np.isin(labels, labels_unique)
                labels = labels[mask_non_single]
                n_clusters_cand = len(set(labels))

                factor = self.compute_clustering_factor(labels, self.take_matrix_by_mask(dist_matrix, mask_non_single))

                clustering_factor[n_clusters_cand] = min(clustering_factor[n_clusters_cand], factor)
                labels_experiment[n_clusters_cand] = labels
                mask_non_single_experiment[n_clusters_cand] = mask_non_single

            n_clusters_best = np.where(clustering_factor < intra_inter_thr_ratio)[0]
            if len(n_clusters_best) > 0:
                n_clusters_best = n_clusters_best[0]
            else:
                n_clusters_best = clustering_factor.argmin()
            print(f"Word {word}: split with n_clusters={n_clusters_best}")
            self.data["train"][word][LABELS_KEY] = labels_experiment[n_clusters_best]
            mask_non_single = mask_non_single_experiment[n_clusters_best]
            self.data["train"][word][TRIALS_KEY] = self.take_trials_by_mask(self.data["train"][word][TRIALS_KEY],
                                                                            mask_non_single)
            self.data["train"][word][INTRA_DIST_KEY] = self.take_matrix_by_mask(dist_matrix, mask_non_single)

            if visualize:
                plt.plot(range(len(clustering_factor)), clustering_factor)
                plt.hlines(y=intra_inter_thr_ratio, xmin=2, xmax=max_clusters, linestyles='dashed')
                plt.scatter(x=n_clusters_best, y=clustering_factor[n_clusters_best], s=30)
                plt.title(f"Word {word}")
                plt.show()
        _save_ujipen(self.data)

    def dbscan(self, scale=0.5, display_outliers=False):
        for word in self.data["train"].keys():
            dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
            eps = scale * dist_matrix.std()
            predictor = DBSCAN(eps=eps, min_samples=2, metric='precomputed', n_jobs=-1)
            labels = predictor.fit_predict(dist_matrix)
            self.data["train"][word][LABELS_KEY] = labels
            self.drop_labels(word, labels_drop=[-1])
            if display_outliers and -1 in labels:
                word_points = self.data["train"][word][TRIALS_KEY]
                display(word_points, labels)

    def display_clustering(self):
        for word in self.data["train"].keys():
            labels = self.data["train"][word].get(LABELS_KEY, None)
            if labels is not None:
                display(self.data["train"][word][TRIALS_KEY], labels)


if __name__ == '__main__':
    ujipen = UJIPen(force_read=True)
    ujipen.dbscan()
    ujipen.cluster(visualize=True)

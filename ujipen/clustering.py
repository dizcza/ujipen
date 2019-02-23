import math
import pickle
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from ujipen.constants import *
from ujipen.loader import ujipen_read, _save_ujipen, filter_alphabet, ujipen_correct_slant, ujipen_normalize, save_intra_dist, \
    check_shapes


class UJIPen:

    def __init__(self, force_read=False):
        if force_read:
            data = ujipen_read()
            filter_alphabet(data)
            ujipen_correct_slant(data)
            ujipen_normalize(data)
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

    @staticmethod
    def cluster_distances(dist_matrix, n_clusters):
        predictor = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                            linkage='average')
        labels = predictor.fit_predict(dist_matrix)
        return labels

    @staticmethod
    def even_clustering(dist_matrix, labels, uneven_size_max_ratio: int):
        assert uneven_size_max_ratio >= 1
        while True:
            labels_unique, counts = np.unique(labels, return_counts=True)
            if len(labels_unique) < 2:
                break
            counts_argsort = np.argsort(counts)[::-1]
            if counts[counts_argsort[1]] == 1:
                break
            if counts[counts_argsort[0]] / counts[counts_argsort[1]] <= uneven_size_max_ratio:
                break
            label_split = labels_unique[counts_argsort[0]]
            mask_split = labels == label_split
            dist_matrix_split = UJIPen.take_matrix_by_mask(dist_matrix, mask_split)
            sublabels = UJIPen.cluster_distances(dist_matrix_split, n_clusters=2)
            sublabels += labels.max() + 1
            labels[mask_split] = sublabels
        return labels

    def cluster(self, max_clusters=10, intra_inter_thr_ratio=0.1, uneven_size_max_ratio: int = 3, visualize=False):
        assert max_clusters >= 2, "At least 2 clusters per word"
        for word in self.data["train"].keys():
            dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
            clustering_factor = defaultdict(lambda: np.inf)
            mask_non_single_experiment = {}
            labels_experiment = {}
            for n_clusters_cand in range(2, max_clusters + 1):
                labels = self.cluster_distances(dist_matrix, n_clusters=n_clusters_cand)
                labels = self.even_clustering(dist_matrix, labels, uneven_size_max_ratio=uneven_size_max_ratio)

                # drop outliers
                labels_unique, counts = np.unique(labels, return_counts=True)
                labels_unique = labels_unique[counts > 1]
                mask_non_single = np.isin(labels, labels_unique)
                labels = labels[mask_non_single]
                n_clusters_cand = len(set(labels))

                factor = self.compute_clustering_factor(labels, self.take_matrix_by_mask(dist_matrix, mask_non_single))

                if factor < clustering_factor[n_clusters_cand]:
                    clustering_factor[n_clusters_cand] = factor
                    labels_experiment[n_clusters_cand] = labels
                    mask_non_single_experiment[n_clusters_cand] = mask_non_single

            n_clusters_best = min(clustering_factor, key=clustering_factor.get)
            for n_clusters_cand in sorted(clustering_factor, key=clustering_factor.get, reverse=True):
                if clustering_factor[n_clusters_cand] < intra_inter_thr_ratio:
                    n_clusters_best = n_clusters_cand
                    break
            print(f"Word {word}: split with n_clusters={n_clusters_best}")
            self.data["train"][word][LABELS_KEY] = labels_experiment[n_clusters_best]
            mask_non_single = mask_non_single_experiment[n_clusters_best]
            self.data["train"][word][TRIALS_KEY] = self.take_trials_by_mask(self.data["train"][word][TRIALS_KEY],
                                                                            mask_non_single)
            self.data["train"][word][INTRA_DIST_KEY] = self.take_matrix_by_mask(dist_matrix, mask_non_single)

            if visualize:
                plt.plot(clustering_factor.keys(), clustering_factor.values())
                plt.hlines(y=intra_inter_thr_ratio, xmin=2, xmax=max_clusters, linestyles='dashed')
                plt.scatter(x=n_clusters_best, y=clustering_factor[n_clusters_best], s=30)
                plt.title(f"Word {word}")
                plt.show()
                self.display(word)
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
                self.display(word)

    def display_clustering(self):
        for word in self.data["train"].keys():
            self.display(word)

    def get_min_intra_dist_patterns(self):
        patterns = {}
        for word in self.data["train"].keys():
            patterns[word] = []
            labels = self.data["train"][word][LABELS_KEY]
            dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
            for label_unique in set(labels):
                mask_cluster = labels == label_unique
                dist_matrix_cluster = self.take_matrix_by_mask(dist_matrix, mask_cluster)
                points_cluster = self.take_trials_by_mask(self.data["train"][word][TRIALS_KEY], mask_cluster)
                best_sample_id = dist_matrix_cluster.sum(axis=0).argmin()
                patterns[word].append(points_cluster[best_sample_id])
        return patterns

    def get_samples(self, fold="train"):
        return {
            word: self.data[fold][word][TRIALS_KEY] for word in self.data[fold].keys()
        }

    def display(self, word, labels=None):
        if labels is None:
            labels = self.data["train"][word][LABELS_KEY]
        word_points = self.data["train"][word][TRIALS_KEY]
        dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
        margin = 0.02
        colors = ['blue', 'orange', 'cyan', 'black', 'magenta']
        rect_size_init = np.array([0.03, 0.03])
        for label in np.unique(labels):
            plt.figure()
            cluster_points = [word_points[i] for i in range(len(word_points)) if labels[i] == label]
            rows = math.floor(math.sqrt(len(cluster_points)))
            cols = math.ceil(len(cluster_points) / rows)
            rect_size = rect_size_init * rows

            dist_matrix_cluster = UJIPen.take_matrix_by_mask(dist_matrix, mask_take=labels == label)
            if dist_matrix is not None:
                min_inter_dist_id = dist_matrix_cluster.sum(axis=0).argmin()
            else:
                min_inter_dist_id = -1

            for i, sample in enumerate(cluster_points):
                sample = sample.copy()
                ax = plt.subplot(rows, cols, i + 1)
                dv = sample[1:] - sample[:-1]
                dist = np.linalg.norm(dv, axis=1)
                corners = np.where(dist > 0.2)[0]
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
    ujipen = UJIPen(force_read=True)
    ujipen.dbscan()
    ujipen.cluster(uneven_size_max_ratio=3, visualize=False)
    ujipen.display_clustering()

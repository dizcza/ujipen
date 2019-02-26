from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from helper import take_matrix_by_mask, take_trials_by_mask, drop_items
from ujipen.loader import _save_ujipen
from ujipen.ujipen_class import UJIPen
from ujipen.ujipen_constants import *


class UJIPenClustering(UJIPen):

    def drop_labels(self, word: str, labels_drop):
        labels = self.data["train"][word][LABELS_KEY]
        dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
        indices_drop = np.where(np.isin(labels, labels_drop))[0]
        if len(indices_drop) == 0:
            return
        self.data["train"][word][TRIALS_KEY] = drop_items(self.data["train"][word][TRIALS_KEY], indices_drop)
        self.data["train"][word][SESSION_KEY] = drop_items(self.data["train"][word][SESSION_KEY], indices_drop)
        self.data["train"][word][LABELS_KEY] = np.delete(labels, indices_drop)
        for axis in (0, 1):
            dist_matrix = np.delete(dist_matrix, indices_drop, axis=axis)
        self.data["train"][word][INTRA_DIST_KEY] = dist_matrix
        _save_ujipen(self.data)
        print(f"Word {word}: dropped {labels_drop}")

    @staticmethod
    def compute_clustering_factor(labels, dist_matrix):
        assert len(labels) == dist_matrix.shape[0] == dist_matrix.shape[1]
        intra_dist = 0
        inter_dist = 0
        for label_unique in set(labels):
            mask = labels == label_unique
            if mask.sum() == 1:
                # skip clusters with a single sample
                continue
            intra_matrix = take_matrix_by_mask(dist_matrix, mask)
            inter_matrix = take_matrix_by_mask(dist_matrix, ~mask)
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
            dist_matrix_split = take_matrix_by_mask(dist_matrix, mask_split)
            sublabels = UJIPenClustering.cluster_distances(dist_matrix_split, n_clusters=2)
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

                factor = self.compute_clustering_factor(labels, take_matrix_by_mask(dist_matrix, mask_non_single))

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
            self.data["train"][word][TRIALS_KEY] = take_trials_by_mask(self.data["train"][word][TRIALS_KEY],
                                                                       mask_non_single)
            self.data["train"][word][SESSION_KEY] = take_trials_by_mask(self.data["train"][word][SESSION_KEY],
                                                                        mask_non_single)
            self.data["train"][word][INTRA_DIST_KEY] = take_matrix_by_mask(dist_matrix, mask_non_single)

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


if __name__ == '__main__':
    ujipen = UJIPenClustering(force_read=True)
    ujipen.dbscan()
    ujipen.cluster(uneven_size_max_ratio=2, visualize=False)
    # ujipen.display_clustering()

import pickle

import matplotlib.pyplot as plt

from helper import take_matrix_by_mask, take_trials_by_mask, display
from ujipen.loader import ujipen_read, _save_ujipen, filter_alphabet, ujipen_correct_slant, ujipen_normalize, \
    save_intra_dist, check_shapes, ujipen_equally_spaced_points
from ujipen.ujipen_constants import *


class UJIPen:

    def __init__(self, force_read=False):
        if force_read:
            data = ujipen_read()
            filter_alphabet(data)
            ujipen_correct_slant(data)
            ujipen_normalize(data)
            ujipen_equally_spaced_points(data)
            save_intra_dist(data)
            _save_ujipen(data, path=UJIPEN_PKL)
        with open(UJIPEN_PKL, 'rb') as f:
            self.data = pickle.load(f)
        check_shapes(self.data)

    @property
    def num_patterns(self):
        patterns = self.get_min_intra_dist_patterns()
        total_patterns = sum(map(len, patterns.values()))
        return total_patterns

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
                dist_matrix_cluster = take_matrix_by_mask(dist_matrix, mask_cluster)
                points_cluster = take_trials_by_mask(self.data["train"][word][TRIALS_KEY], mask_cluster)
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
        display(word_points, labels=labels, dist_matrix=dist_matrix)

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
    ujipen.show_trial_size()
    # ujipen.display_clustering()

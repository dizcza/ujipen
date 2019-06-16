import math
import pickle

import matplotlib.pyplot as plt
import numpy as np

from helper import draw_sample, create_edge_rectangles_patch
from ujipen.loader import ujipen_read, filter_alphabet, ujipen_correct_slant, ujipen_normalize, check_shapes, \
    ujipen_drop_from_dropped_list
from ujipen.ujipen_constants import *


class UJIPen:

    def __init__(self, force_read=False):
        if force_read:
            data = ujipen_read()
            filter_alphabet(data)
            ujipen_correct_slant(data)
            ujipen_normalize(data)
            self.data = data
            self.save()
        with open(UJIPEN_PKL, 'rb') as f:
            self.data = pickle.load(f)
        ujipen_drop_from_dropped_list(self.data)
        check_shapes(self.data)

    def save(self):
        check_shapes(self.data)
        with open(UJIPEN_PKL, 'wb') as f:
            pickle.dump(self.data, f)

    def display_clustering(self):
        for word in self.data["train"].keys():
            self.display(word)

    def get_samples(self, fold="train"):
        return {
            word: self.data[fold][word][TRIALS_KEY] for word in self.data[fold].keys()
        }

    def display(self, word: str):
        word_points = self.data["train"][word][TRIALS_KEY]
        sample_ids = self.data["train"][word][SESSION_KEY]
        rect_size_init = np.array([0.03, 0.03])
        rows = math.floor(math.sqrt(len(word_points)))
        cols = math.ceil(len(word_points) / rows)
        rect_size = rect_size_init * rows

        for i, sample in enumerate(word_points):
            ax = plt.subplot(rows, cols, i + 1)
            draw_sample(sample)
            ax.add_collection(create_edge_rectangles_patch(sample, rect_size=rect_size))
            plt.title(sample_ids[i], fontsize=5)
        plt.suptitle(f"Word '{word}'")
        plt.show()

    def show_strokes_count_hist(self):
        strokes_count = []
        samples = self.get_samples()
        for word, trials in samples.items():
            strokes_count.extend(map(len, trials))
        plt.hist(strokes_count)
        plt.xlabel("# of strokes in a word")
        plt.ylabel("# of data points that have such a num. of strokes")
        plt.xticks(ticks=range(1, 1 + max(strokes_count)))
        plt.title("Distribution of num. of strokes per word")
        plt.show()


if __name__ == '__main__':
    ujipen = UJIPen(force_read=True)
    ujipen.show_strokes_count_hist()
    # ujipen.display_clustering()

import math
import pickle
import string

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import warnings

from ujipen.constants import *
from ujipen.loader import read_ujipen, _save_ujipen, filter_alphabet, correct_slant, normalize, save_intra_dist


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

    def drop_labels(self, word: str, labels_drop):
        word_points = self.data["train"][word][TRIALS_KEY]
        labels = self.data["train"][word][LABELS_KEY]
        dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
        assert len(word_points) == len(labels)
        word_points = [word_points[i] for i in range(len(word_points)) if labels[i] not in labels_drop]
        indices_drop = np.where(np.isin(labels, labels_drop))[0]
        labels = np.delete(labels, indices_drop)
        for axis in (0, 1):
            dist_matrix = np.delete(dist_matrix, indices_drop, axis=axis)
        assert len(word_points) == len(labels) == dist_matrix.shape[0] == dist_matrix.shape[1]
        self.data["train"][word][TRIALS_KEY] = word_points
        self.data["train"][word][LABELS_KEY] = labels
        self.data["train"][word][INTRA_DIST_KEY] = dist_matrix
        _save_ujipen(self.data)
        _save_ujipen(self.data, path=UJIPEN_DBSCANNED)
        print(f"Word {word}: dropped {labels_drop}")

    def cluster(self, command, dist_matrix, labels,):
        if 'label' in command:
            label = int(command[command.index('label') + len('label=')])
            indices_drop = np.where(labels == label)[0]
            dist_matrix_label = dist_matrix.copy()
            for axis in (0, 1):
                dist_matrix_label = np.delete(dist_matrix_label, indices_drop, axis=axis)
            predictor = DBSCAN(eps=dist_matrix_label.std() / scale, min_samples=2, metric='precomputed', n_jobs=-1)
            sublabels = predictor.fit_predict(dist_matrix_label)
            sublabels += labels.max() + 2
            labels[indices_drop] = sublabels
        else:
            predictor = DBSCAN(eps=dist_matrix.std() / scale, min_samples=2, metric='precomputed', n_jobs=-1)
            labels = predictor.fit_predict(dist_matrix)

    def dbscan(self, word: str):
        command = ''
        while command != 'next':
            word_points = self.data["train"][word][TRIALS_KEY]
            dist_matrix = self.data["train"][word][INTRA_DIST_KEY]
            labels = self.data["train"][word].get(LABELS_KEY, np.zeros(len(word_points), dtype=int))
            display(word_points, labels)
            command = input('Next?\n').lower()
            if command.startswith('dbscan'):
                scale = int(command[len('dbscan ')])
                if 'label' in command:
                    label = int(command[command.index('label') + len('label=')])
                    indices_leave = np.where(labels == label)[0]
                    dist_matrix_label = dist_matrix.copy()
                    for axis in (0, 1):
                        dist_matrix_label = np.delete(dist_matrix_label, indices_leave, axis=axis)
                    predictor = DBSCAN(eps=dist_matrix_label.std() / scale, min_samples=2, metric='precomputed', n_jobs=-1)
                    sublabels = predictor.fit_predict(dist_matrix_label)
                    sublabels += labels.max() + 2
                    labels[indices_leave] = sublabels
                else:
                    predictor = DBSCAN(eps=dist_matrix.std() / scale, min_samples=2, metric='precomputed', n_jobs=-1)
                    labels = predictor.fit_predict(dist_matrix)
                self.data["train"][word][LABELS_KEY] = labels
            elif command.startswith('cluster'):
                n_clusters = int(command[len('cluster ')])
                if 'linkage' in command:
                    linkage = command[command.index('linkage') + len('linkage='):]
                else:
                    linkage = 'single'

                message = f"clustering with n_clusters={n_clusters}, linkage={linkage}"

                if 'label' in command:
                    label = int(command[command.index('label') + len('label=')])
                    message += f' label={label}'
                    indices_leave = np.where(labels == label)[0]
                    dist_matrix_label = dist_matrix.copy()
                    for axis in (0, 1):
                        dist_matrix_label = np.delete(dist_matrix_label, indices_leave, axis=axis)
                    predictor = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                                        linkage=linkage)
                    sublabels = predictor.fit_predict(dist_matrix_label)
                    sublabels += labels.max() + 2
                    labels[indices_leave] = sublabels
                else:
                    predictor = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                                        linkage=linkage)
                    labels = predictor.fit_predict(dist_matrix)
                print(message)
                self.data["train"][word][LABELS_KEY] = labels
            elif command.startswith('drop'):
                labels_drop = command[len('drop '):].split(' ')
                labels_drop = np.asarray(labels_drop, dtype=int)
                self.drop_labels(word=word, labels_drop=labels_drop)


if __name__ == '__main__':
    ujipen = UJIPen()
    for word in string.ascii_lowercase[3:]:
        ujipen.dbscan(word)

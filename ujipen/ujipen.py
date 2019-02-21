from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import math
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import string
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from dtw_solver import dtw_vectorized

UJIPEN_DIR = Path.home() / "dataset" / "ujipenchars2"
UJIPEN_PATH = UJIPEN_DIR / "ujipenchars2.txt"
UJIPEN_PATH_PICKLED = UJIPEN_PATH.with_suffix('.pkl')

INTRA_DIST_KEY = "intra-dist"


def _save_ujipen(data):
    with open(UJIPEN_PATH_PICKLED, 'wb') as f:
        pickle.dump(data, f)


def read_ujipen(filter_duplicates=True):
    if UJIPEN_PATH_PICKLED.exists():
        print(f'Using pickled {UJIPEN_PATH_PICKLED}')
        with open(UJIPEN_PATH_PICKLED, 'rb') as f:
            data = pickle.load(f)
        return data
    with open(UJIPEN_PATH) as f:
        lines = f.readlines()
    data = {
        fold: {} for fold in ('train', 'test')
    }
    line_id = 0
    while line_id < len(lines):
        line = lines[line_id]
        if line.startswith('WORD'):
            word = line[5]
            if line[7:].startswith('trn'):
                fold = "train"
            else:
                fold = "test"
            if word not in data[fold]:
                data[fold][word] = {"trials": []}
            line_id += 1
            numstrokes = int(lines[line_id][13:])
            assert numstrokes <= 5
            points = []
            for stroke_id in range(numstrokes):
                line_id += 1
                line = lines[line_id]
                hashtag_pos = line.index('#')
                num_points = int(line[9: hashtag_pos - 1])
                stroke_points = line[hashtag_pos + 2:].split(' ')
                stroke_points = np.asarray(stroke_points, dtype=np.float32).reshape(-1, 2)
                assert len(stroke_points) == num_points
                if filter_duplicates:
                    duplicates = (stroke_points[1:] == stroke_points[:-1]).all(axis=1)
                    duplicates = np.r_[False, duplicates]  # always add first
                    stroke_points = stroke_points[~duplicates]
                points.append(stroke_points)
            points = np.vstack(points)  # todo handle strokes
            data[fold][word]["trials"].append(points)
        line_id += 1
    _save_ujipen(data)
    return data


def _corr_sl(points, vert_slant_angle=50):
    vert_slant_angle = math.radians(vert_slant_angle)
    vert_slant_cotang = 1 / math.tan(vert_slant_angle)

    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    take = np.abs(dy / dx) > vert_slant_cotang
    dx = dx[take]
    dy = dy[take]
    dx[dy < 0] *= -1
    dy[dy < 0] *= -1

    slant_dx, slant_dy = np.c_[dx, dy].sum(axis=0)
    if slant_dy == 0:
        return
    shear = -slant_dx / slant_dy
    print(shear)
    points[:, 0] = points[:, 0] + points[:, 1] * shear


def correct_slant(data, vert_slant_angle=50):
    vert_slant_angle = math.radians(vert_slant_angle)
    vert_slant_cotang = 1 / math.tan(vert_slant_angle)
    for fold in data.keys():
        for word, trials in data[fold].items():
            for points in trials["trials"]:
                dx = np.diff(points[:, 0])
                dy = np.diff(points[:, 1])
                take = np.abs(dy / dx) > vert_slant_cotang
                dx = dx[take]
                dy = dy[take]
                dx[dy < 0] *= -1
                dy[dy < 0] *= -1

                slant_dx, slant_dy = np.c_[dx, dy].sum(axis=0)
                if slant_dy == 0:
                    continue
                shear = -slant_dx / slant_dy
                points[:, 0] = points[:, 0] + points[:, 1] * shear


def normalize(data, keep_aspect_ratio=True):
    eps = 1e-6
    for fold in data.keys():
        for word, trials in data[fold].items():
            for points in trials["trials"]:
                x, y = points.T
                ymin, xmin, ymax, xmax = y.min(), x.min(), y.max(), x.max()
                h = ymax - ymin
                w = xmax - xmin
                scale_x = 1 / w
                scale_y = 1 / h
                if keep_aspect_ratio:
                    scale_x = scale_y = min(scale_x, scale_y)
                xc = (xmin + xmax) / 2
                yc = (ymin + ymax) / 2
                points[:, 0] = 0.5 + (x - xc) * scale_x
                points[:, 1] = 0.5 + (y - yc) * scale_y
                assert np.logical_and(points >= -eps, points <= 1. + eps).all()


def filter_alphabet(data, alphabet=string.ascii_lowercase):
    for fold in data.keys():
        data[fold] = {
            word: data[fold][word] for word in alphabet
        }


def display(points, labels, margin=0.02):
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
            for chunk in np.split(sample, corners):
                x, y = chunk.T
                plt.plot(x, y)
            rects = [mpatches.Rectangle(sample[pid] - rect_size / 2, *rect_size) for pid in (0, -1)]
            pc = PatchCollection(rects, facecolors=['g', 'r'])
            ax.add_collection(pc)
            plt.xlim(left=0 - margin, right=1 + margin)
            plt.ylim(bottom=0 - margin, top=1 + margin)
            plt.axis('off')
        plt.suptitle(f'Label {label}')
    plt.show()


def save_intra_dist(data):
    for word, trials in data["train"].items():
        if INTRA_DIST_KEY in data["train"][word]:
            continue
        word_points = trials["trials"]
        dist_matrix = np.zeros((len(word_points), len(word_points)), dtype=np.float32)
        for i, anchor in enumerate(tqdm(word_points, desc=f"Word {word}")):
            for j, sample in enumerate(word_points[i + 1:], start=i + 1):
                dist = dtw_vectorized(sample, anchor)[-1, -1]
                dist /= len(sample) + len(anchor)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        data["train"][word][INTRA_DIST_KEY] = dist_matrix
    _save_ujipen(data)


def clean(data, word='a'):
    def drop_labels(labels, labels_drop):
        assert len(labels) == len(word_points)
        word_points_filtered = []
        labels_filtered = []
        indices_drop = np.where(np.isin(labels_drop, labels))[0]
        dist_matrix_filtered = np.delete(dist_matrix, indices_drop, axis=0)
        dist_matrix_filtered = np.delete(dist_matrix_filtered, indices_drop, axis=1)
        for points, label in zip(word_points, labels):
            if label not in labels_drop:
                word_points_filtered.append(points)
                labels_filtered.append(label)
        return word_points_filtered, labels_filtered, dist_matrix_filtered

    dist_matrix = data["train"][word][INTRA_DIST_KEY]
    word_points = data["train"][word]["trials"]
    predictor = DBSCAN(eps=dist_matrix.std() / 2, min_samples=2, metric='precomputed', n_jobs=-1)
    labels = predictor.fit_predict(dist_matrix)
    print(labels)
    display(word_points, labels)

    command = input("Next?\n").lower()
    while command != 'next':
        if command.startswith('drop'):
            labels_drop = command[len('drop '):].split(' ')
            labels_drop = np.asarray(labels_drop, dtype=int)
            word_points, labels, dist_matrix = drop_labels(labels=labels, labels_drop=labels_drop)
            print(f"Dropped {labels_drop}")
            predictor = DBSCAN(eps=dist_matrix.std() / 2, min_samples=2, metric='precomputed', n_jobs=-1)
            labels = predictor.fit_predict(dist_matrix)
        elif command.startswith('cluster'):
            n_clusters = int(command[len('cluster ')])
            if 'linkage' in command:
                linkage = command[command.index('linkage') + len('linkage '):]
            else:
                linkage = 'single'
            print(f"clustering with n_clusters={n_clusters}, linkage={linkage}")
            predictor = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage=linkage)
            labels = predictor.fit_predict(dist_matrix)
        display(word_points, labels)
        command = input('Next?\n').lower()
    data["train"][word]["clusters"] = labels
    _save_ujipen(data)


if __name__ == '__main__':
    data = read_ujipen()
    filter_alphabet(data)
    correct_slant(data)
    normalize(data)
    save_intra_dist(data)
    for word in string.ascii_lowercase:
        clean(data, word)

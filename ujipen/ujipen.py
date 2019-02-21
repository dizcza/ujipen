from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import math
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import string
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from dtw_solver import dtw_vectorized

UJIPEN_PATH = Path.home() / "dataset" / "ujipenchars2" / "ujipenchars2.txt"
UJIPEN_PATH_PICKLED = UJIPEN_PATH.with_suffix('.pkl')


def read_ujipen(filter_duplicates=True):
    if UJIPEN_PATH_PICKLED.exists():
        print(f'Using pickled {UJIPEN_PATH_PICKLED}')
        with open(UJIPEN_PATH_PICKLED, 'rb') as f:
            data = pickle.load(f)
        return data
    with open(UJIPEN_PATH) as f:
        lines = f.readlines()
    data = {
        "train": defaultdict(list),
        "test": defaultdict(list)
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
            data[fold][word].append(points)
        line_id += 1
    with open(UJIPEN_PATH_PICKLED, 'wb') as f:
        pickle.dump(data, f)
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
            for points in trials:
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
            for points in trials:
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
    for label in np.unique(labels):
        plt.figure()
        label_points = [points[i] for i in range(len(points)) if labels[i] == label]
        rows = math.floor(math.sqrt(len(label_points)))
        cols = math.ceil(len(label_points) / rows)
        for i, sample in enumerate(label_points, start=1):
            sample = sample.copy()
            plt.subplot(rows, cols, i)
            dv = sample[1:] - sample[:-1]
            dist = np.linalg.norm(dv, axis=1)
            corners = np.where(dist > 0.2)[0]
            corners += 1
            sample[:, 1] = sample[:, 1].max() - sample[:, 1]
            for chunk in np.split(sample, corners):
                x, y = chunk.T
                plt.plot(x, y)
            plt.xlim(left=0 - margin, right=1 + margin)
            plt.ylim(bottom=0 - margin, top=1 + margin)
            plt.axis('off')
        plt.suptitle(f'Label {label}')
    plt.show()


def visualize_inner_dist(data, word='a'):
    word_points = data["train"][word]
    # word_points = word_points[:40]
    dist_matrix = np.zeros((len(word_points), len(word_points)), dtype=np.float32)
    for i, anchor in enumerate(tqdm(word_points)):
        for j, sample in enumerate(word_points[i + 1:], start=i + 1):
            dist = dtw_vectorized(sample, anchor)[-1, -1]
            dist /= len(sample) + len(anchor)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    nondiag_dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    print(f"Word '{word}' inner dist: {nondiag_dist.mean():.5f} ± {nondiag_dist.std():.5f} "
          f"(min={nondiag_dist.min():.5f}, max={nondiag_dist.max():.5f})")

    predictor = DBSCAN(eps=dist_matrix.std() / 2, min_samples=2, metric='precomputed', n_jobs=-1)
    labels = predictor.fit_predict(dist_matrix)
    print(labels)

    display(word_points, labels)

    tsne = TSNE(n_components=2, metric='precomputed')
    transformed = tsne.fit_transform(dist_matrix)
    plt.scatter(transformed[:, 0], transformed[:, 1])
    plt.show()

    plt.hist(nondiag_dist)
    plt.show()


if __name__ == '__main__':
    data = read_ujipen()
    filter_alphabet(data)
    # del data['test']
    # data['train'] = {'a': data['train']['a']}
    correct_slant(data)
    normalize(data)
    for word in string.ascii_lowercase:
        visualize_inner_dist(data, word)

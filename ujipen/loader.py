import math
import pickle
import string
import warnings

import numpy as np
import requests
from tqdm import tqdm

from dtw_solver import dtw_vectorized

from ujipen.constants import *


def _save_ujipen(data, path=UJIPEN_PKL):
    check_shapes(data)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def download_ujipen():
    UJIPEN_DIR.mkdir(parents=True, exist_ok=True)
    request = requests.get(UJIPEN_URL, stream=True)
    total_size = int(request.headers.get('content-length', 0))
    wrote_bytes = 0
    with open(UJIPEN_TXT, 'wb') as f:
        for data in tqdm(request.iter_content(chunk_size=1024), desc=f"Downloading {UJIPEN_URL}",
                         total=total_size // 1024,
                         unit='KB', unit_scale=True):
            wrote_bytes += f.write(data)
    if wrote_bytes != total_size:
        warnings.warn("Content length mismatch. Try downloading again.")


def check_shapes(data):
    for word in data["train"].keys():
        word_points = data["train"][word][TRIALS_KEY]
        dist_matrix = data["train"][word].get(INTRA_DIST_KEY, None)
        shapes = [len(word_points)]
        if dist_matrix is not None:
            shapes.extend(dist_matrix.shape)
        labels = data["train"][word].get(LABELS_KEY, None)
        if labels is not None:
            shapes.append(len(labels))
        assert len(set(shapes)) == 1  # all equal


def read_ujipen(filter_duplicates=True):
    if not UJIPEN_TXT.exists():
        download_ujipen()
    with open(UJIPEN_TXT) as f:
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
                data[fold][word] = {TRIALS_KEY: []}
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
            data[fold][word][TRIALS_KEY].append(points)
        line_id += 1
    return data


def correct_slant(data):
    vert_slant_radians = math.radians(VERT_SLANGE_ANGLE)
    vert_slant_cotang = 1 / math.tan(vert_slant_radians)
    for fold in data.keys():
        for word, trials in data[fold].items():
            for points in trials[TRIALS_KEY]:
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
            for points in trials[TRIALS_KEY]:
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


def save_intra_dist(data):
    intra_dist = {}  # todo remove this hack
    if UJIPEN_INTRA_DIST_PATH.exists():
        with open(UJIPEN_INTRA_DIST_PATH, 'rb') as f:
            intra_dist = pickle.load(f)
        for word in data["train"].keys():
            data["train"][word][INTRA_DIST_KEY] = intra_dist[word]
        return

    for word, trials in data["train"].items():
        if INTRA_DIST_KEY in data["train"][word]:
            continue
        word_points = trials[TRIALS_KEY]
        dist_matrix = np.zeros((len(word_points), len(word_points)), dtype=np.float32)
        for i, anchor in enumerate(tqdm(word_points, desc=f"Word {word}")):
            for j, sample in enumerate(word_points[i + 1:], start=i + 1):
                dist = dtw_vectorized(sample, anchor)[-1, -1]
                dist /= len(sample) + len(anchor)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        data["train"][word][INTRA_DIST_KEY] = dist_matrix
        intra_dist[word] = dist_matrix
    _save_ujipen(data)
    with open(UJIPEN_INTRA_DIST_PATH, 'wb') as f:
        pickle.dump(intra_dist, f)

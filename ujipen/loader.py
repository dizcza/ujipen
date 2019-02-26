import pickle
import string
import warnings

import numpy as np
import requests
from tqdm import tqdm

from dtw_solver import dtw_vectorized
from preprocess import normalize, correct_slant, filter_duplicates, equally_spaced_points
from ujipen.ujipen_constants import *

from helper import drop_items


def _save_ujipen(data, path=UJIPEN_PKL):
    check_shapes(data)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def ujipen_download():
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
        sample_ids = data["train"][word][SESSION_KEY]
        dist_matrix = data["train"][word].get(INTRA_DIST_KEY, None)
        shapes = [len(word_points), len(sample_ids)]
        if dist_matrix is not None:
            shapes.extend(dist_matrix.shape)
        labels = data["train"][word].get(LABELS_KEY, None)
        if labels is not None:
            shapes.append(len(labels))
        assert len(set(shapes)) == 1  # all equal


def ujipen_read():
    if not UJIPEN_TXT.exists():
        ujipen_download()
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
            sample_id = f"{word} {line[7:-1]}"
            if line[7:].startswith('trn'):
                fold = "train"
            else:
                fold = "test"
            if word not in data[fold]:
                data[fold][word] = {
                    TRIALS_KEY: [],
                    SESSION_KEY: []
                }
            data[fold][word][SESSION_KEY].append(sample_id)
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
                stroke_points = filter_duplicates(stroke_points)
                points.append(stroke_points)
            points = np.vstack(points)  # todo handle strokes
            data[fold][word][TRIALS_KEY].append(points)
        line_id += 1
    return data


def ujipen_drop_from_dropped_list(data):
    dropped_list = []
    if UJIPEN_DROPPED_LIST.exists():
        with open(UJIPEN_DROPPED_LIST) as f:
            dropped_list = f.read().splitlines()
    dropped = {
        "train": set(filter(lambda sample_id: 'trn' in sample_id, dropped_list)),
        "test": set(filter(lambda sample_id: 'tst' in sample_id, dropped_list))
    }
    for fold in data.keys():
        for word, trials in data[fold].items():
            sample_ids = trials[SESSION_KEY]
            drop_ids = [i for i in range(len(sample_ids)) if sample_ids[i] in dropped[fold]]
            if len(drop_ids) == 0:
                continue
            trials[TRIALS_KEY] = drop_items(trials[TRIALS_KEY], drop_ids)
            trials[SESSION_KEY] = drop_items(sample_ids, drop_ids)
            labels = trials.get(LABELS_KEY, None)
            if labels is not None:
                trials[LABELS_KEY] = np.delete(labels, drop_ids)
            dist_matrix = trials.get(INTRA_DIST_KEY, None)
            if dist_matrix is not None:
                for axis in (0, 1):
                    dist_matrix = np.delete(dist_matrix, drop_ids, axis=axis)
                trials[INTRA_DIST_KEY] = dist_matrix
    check_shapes(data)
    print(f"Dropped {len(dropped_list)} manually selected samples.")


def ujipen_correct_slant(data):
    for fold in data.keys():
        for word, trials in data[fold].items():
            for points in trials[TRIALS_KEY]:
                correct_slant(points)


def ujipen_normalize(data, keep_aspect_ratio=True):
    for fold in data.keys():
        for word, trials in data[fold].items():
            for points in trials[TRIALS_KEY]:
                normalize(points, keep_aspect_ratio=keep_aspect_ratio)


def ujipen_equally_spaced_points(data):
    for fold in data.keys():
        samples = {
            word: trials[TRIALS_KEY] for word, trials in data[fold].items()
        }
        samples_equally_spaced = equally_spaced_points(samples)
        for word in data[fold].keys():
            data[fold][word][TRIALS_KEY] = samples_equally_spaced[word]


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

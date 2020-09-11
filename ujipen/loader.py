import string
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm

from helper import drop_items
from preprocess import normalize, correct_slant, filter_duplicates, \
    equally_spaced_points_patterns
from ujipen.ujipen_constants import *


class TqdmUpTo(tqdm):
    """
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Original implementation:
    https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b : int, optional
            Number of blocks transferred so far [default: 1].
        bsize : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def ujipen_download(verbose=True):
    UJIPEN_DIR.mkdir(parents=True, exist_ok=True)
    desc = f"Downloading {UJIPEN_URL} to '{UJIPEN_TXT}'"
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=desc, disable=not verbose) as t:
        urlretrieve(UJIPEN_URL, filename=UJIPEN_TXT, reporthook=t.update_to)


def check_shapes(data):
    for word in data["train"].keys():
        word_points = data["train"][word][TRIALS_KEY]
        sample_ids = data["train"][word][SESSION_KEY]
        shapes = [len(word_points), len(sample_ids)]
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
                points.append(stroke_points)
            points = filter_duplicates(points)
            data[fold][word][TRIALS_KEY].append(points)
        line_id += 1
    check_shapes(data)
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
        samples_equally_spaced = equally_spaced_points_patterns(samples)
        for word in data[fold].keys():
            data[fold][word][TRIALS_KEY] = samples_equally_spaced[word]


def filter_alphabet(data, alphabet=string.ascii_lowercase):
    for fold in data.keys():
        data[fold] = {
            word: data[fold][word] for word in alphabet
        }


if __name__ == '__main__':
    ujipen_download()

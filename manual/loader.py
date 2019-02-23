import numpy as np

from manual.constants import POINTS_DIR
from preprocess import normalize, filter_duplicates


def load_manual_patterns():
    data = {}
    for filepath in sorted(POINTS_DIR.iterdir()):
        word = filepath.stem
        sample_points = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
        sample_points = filter_duplicates(sample_points)
        normalize(sample_points)
        data[word] = [sample_points]
    return data

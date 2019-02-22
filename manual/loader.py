import numpy as np

from manual.constants import POINTS_DIR


def load_manual_patterns():
    data = {}
    for filepath in sorted(POINTS_DIR.iterdir()):
        word = filepath.stem
        data[word] = [np.loadtxt(filepath, dtype=np.int32, delimiter=',')]
    return data

from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

from dtw_solver import dtw_vectorized
from ujipen.clustering import UJIPen

from ujipen.constants import UJIPEN_DIR

UJIPEN_PRECOMPUTED_DISTANCES_PATH = UJIPEN_DIR / "distances.pkl"


def compute_distances(sample, patterns):
    distances = {}
    for word, pattern_trials in patterns.items():
        distances[word] = [dtw_vectorized(sample, pattern)[-1, -1] for pattern in pattern_trials]
    return distances


def save_distances(fold="train"):
    ujipen = UJIPen()
    patterns = ujipen.get_min_intra_dist_patterns()
    samples = ujipen.get_samples(fold=fold)
    distances = {}
    if UJIPEN_PRECOMPUTED_DISTANCES_PATH.exists():
        with open(UJIPEN_PRECOMPUTED_DISTANCES_PATH, 'rb') as f:
            distances = pickle.load(f)
    distances[fold] = {}
    for word, sample_trials in samples.items():
        distances[fold][word] = []
        for sample in tqdm(sample_trials, desc=f"Computing distances {fold} word '{word}'"):
            sample_distances = compute_distances(sample, patterns)
            distances[fold][word].append(sample_distances)
    with open(UJIPEN_PRECOMPUTED_DISTANCES_PATH, 'wb') as f:
        pickle.dump(distances, f)


def classify(sample, patterns):
    distances = compute_distances(sample, patterns)
    for word in distances.keys():
        distances[word] = min(distances[word])
    predicted = min(distances, key=distances.get)
    return predicted


def test(fold="train", use_cached=True):
    if use_cached:
        with open(UJIPEN_PRECOMPUTED_DISTANCES_PATH, 'rb') as f:
            distances = pickle.load(f)[fold]
        labels_true = []
        labels_predicted = []
        for word in distances.keys():
            labels_true.extend([word] * len(distances[word]))
            for sample_dists in distances[word]:
                predicted = min(sample_dists, key=lambda word_pred: min(sample_dists[word_pred]))
                labels_predicted.append(predicted)
    else:
        ujipen = UJIPen()
        patterns = ujipen.get_min_intra_dist_patterns()
        samples = ujipen.get_samples(fold=fold)
        labels_true = []
        labels_predicted = []
        for word, sample_trials in samples.items():
            labels_true.extend([word] * len(sample_trials))
            for sample in tqdm(sample_trials, desc=f"Predicting {fold} word '{word}'"):
                predicted = classify(sample, patterns)
                labels_predicted.append(predicted)
    labels_true = np.asarray(labels_true)
    labels_predicted = np.asarray(labels_predicted)
    accuracy = np.mean(labels_true == labels_predicted)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(labels_true, labels_predicted)
    plt.imshow(cm)
    plt.show()


if __name__ == '__main__':
    save_distances("train")
    test(use_cached=True)

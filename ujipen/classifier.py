import pickle
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
import concurrent

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from dtw_solver import dtw_vectorized
from helper import draw_sample
from ujipen.ujipen_class import UJIPen
from ujipen.ujipen_constants import UJIPEN_DIR

UJIPEN_PRECOMPUTED_DISTANCES_PATH = UJIPEN_DIR / "distances.pkl"


def _compute_distance_background(sample, pattern, word_pattern: str):
    dist = dtw_vectorized(sample=sample, pattern=pattern)[-1, -1]
    return dist, word_pattern


def compute_distances(sample: List[np.ndarray], patterns: Dict[str, List[List[np.ndarray]]]) -> Dict[str, List[float]]:
    sample = np.vstack(sample)
    distances_to_patterns = {}
    futures_list = []
    with ProcessPoolExecutor() as executor:
        for word_pattern, pattern_trials in patterns.items():
            distances_to_patterns[word_pattern] = []
            for pattern in pattern_trials:
                pattern = np.vstack(pattern)
                future = executor.submit(_compute_distance_background, sample=sample, pattern=pattern,
                                         word_pattern=word_pattern)
                futures_list.append(future)
    for future in concurrent.futures.as_completed(futures_list):
        dist, word_pattern = future.result()
        distances_to_patterns[word_pattern].append(dist)
    return distances_to_patterns


def save_distances(fold="train"):
    ujipen = UJIPen()
    patterns = ujipen.get_min_intra_dist_patterns()
    samples = ujipen.get_samples(fold=fold)
    distances = {}
    if UJIPEN_PRECOMPUTED_DISTANCES_PATH.exists():
        with open(UJIPEN_PRECOMPUTED_DISTANCES_PATH, 'rb') as f:
            distances = pickle.load(f)
    distances[fold] = {}
    for word_sample, sample_trials in samples.items():
        distances[fold][word_sample] = []
        for sample in tqdm(sample_trials, desc=f"Computing distances {fold} word '{word_sample}'"):
            sample_distances = compute_distances(sample, patterns)
            distances[fold][word_sample].append(sample_distances)
        break
    with open(UJIPEN_PRECOMPUTED_DISTANCES_PATH, 'wb') as f:
        pickle.dump(distances, f)


def classify(sample: List[np.ndarray], patterns: Dict[str, List[List[np.ndarray]]], k_neighbors: int) -> str:
    distances = compute_distances(sample, patterns)
    labels = []
    distances_ravel = []
    for word in distances.keys():
        labels.extend([word] * len(distances[word]))
        distances_ravel.extend(distances[word])
    argsort = np.argsort(distances_ravel)
    label_majority = majority_label(labels=labels, neighbor_ids=argsort[:k_neighbors])
    return label_majority


def majority_label(labels, neighbor_ids):
    labels_neighbors = np.take(labels, neighbor_ids)
    labels_neighbors_unique, counts = np.unique(labels_neighbors, return_counts=True)
    label_majority = labels_neighbors_unique[counts.argmax()]
    return label_majority


def knn_precomputed(distances_to_patterns: Dict[str, List[float]], k_neighbors: int) -> str:
    labels = []
    dists = []
    for word, word_dists in distances_to_patterns.items():
        labels.extend([word] * len(word_dists))
        dists.extend(word_dists)
    argsort = np.argsort(dists)
    label_majority = majority_label(labels=labels, neighbor_ids=argsort[:k_neighbors])
    return label_majority


def test_knn_correctness():
    with open(UJIPEN_PRECOMPUTED_DISTANCES_PATH, 'rb') as f:
        distances = pickle.load(f)["train"]
    for word_sample in distances.keys():
        for sample_dists in distances[word_sample]:
            closest_label = min(sample_dists, key=lambda word_pred: min(sample_dists[word_pred]))
            knn_predicted = knn_precomputed(sample_dists, k_neighbors=1)
            assert knn_predicted == closest_label


def show_knn(k_neighbors=3):
    ujipen = UJIPen()
    patterns = ujipen.get_min_intra_dist_patterns()
    samples = ujipen.get_samples(fold="train")
    for word_sample, sample_trials in samples.items():
        for sample in sample_trials:
            distances_to_patterns = compute_distances(sample, patterns)
            distances_ravel = []
            patterns_ravel = []
            patterns_words_ravel = []
            for word_pattern, word_dists in distances_to_patterns.items():
                distances_ravel.extend(word_dists)
                patterns_ravel.extend(patterns[word_pattern])
                patterns_words_ravel.extend([word_pattern] * len(word_dists))
            argsort = np.argsort(distances_ravel)
            plt.figure(figsize=(3 * k_neighbors, 4.8))
            plt.subplot(1, k_neighbors + 1, 1)
            draw_sample(sample)
            plt.title('Sample')
            for axes_id, pattern_id in enumerate(argsort[:k_neighbors], start=2):
                plt.subplot(1, k_neighbors + 1, axes_id)
                draw_sample(patterns_ravel[pattern_id])
                plt.title(f"{patterns_words_ravel[pattern_id]} {distances_ravel[pattern_id]:.5f}")
            plt.show()


def test(fold="train", k_neighbors=5, use_cached=True):
    ujipen = UJIPen()
    samples = ujipen.get_samples(fold=fold)
    if use_cached:
        with open(UJIPEN_PRECOMPUTED_DISTANCES_PATH, 'rb') as f:
            distances = pickle.load(f)[fold]
        labels_true = []
        labels_predicted = []
        for word_sample in distances.keys():
            assert len(samples[word_sample]) == len(distances[word_sample])
            labels_true.extend([word_sample] * len(distances[word_sample]))
            for distances_to_patterns in distances[word_sample]:
                closest_label = knn_precomputed(distances_to_patterns, k_neighbors=k_neighbors)
                labels_predicted.append(closest_label)
    else:
        patterns = ujipen.get_min_intra_dist_patterns()
        labels_true = []
        labels_predicted = []
        for word_sample, sample_trials in samples.items():
            labels_true.extend([word_sample] * len(sample_trials))
            for sample in tqdm(sample_trials, desc=f"Predicting {fold} word '{word_sample}'"):
                closest_label = classify(sample, patterns, k_neighbors=k_neighbors)
                labels_predicted.append(closest_label)
    labels_true = np.asarray(labels_true)
    labels_predicted = np.asarray(labels_predicted)
    accuracy = np.mean(labels_true == labels_predicted)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(labels_true, labels_predicted)
    plt.imshow(cm)
    plt.show()


def show_distances_hist(k_neighbors=5):
    with open(UJIPEN_PRECOMPUTED_DISTANCES_PATH, 'rb') as f:
        distances = pickle.load(f)["train"]
    distances_same_word = []
    for word_sample in distances.keys():
        for distances_to_patterns in distances[word_sample]:
            closest_label = knn_precomputed(distances_to_patterns, k_neighbors=k_neighbors)
            distances_same_word.append(distances_to_patterns[closest_label])
    distances = np.hstack(distances_same_word)
    distances = distances[distances > 0]
    distances = np.sort(distances)
    print(f"DTW sum distances: {distances.mean():.3f} +- {distances.std():.3f} "
          f"(min={distances.min()}, max={distances.max()})")
    plt.hist(distances, bins=100)
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    # save_distances("train")
    show_distances_hist()
    quit()
    # quit()
    # test_knn_correctness()
    show_knn(k_neighbors=5)
    # test(use_cached=True, k_neighbors=1)

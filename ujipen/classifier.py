import concurrent
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from dtw_solver import dtw_vectorized
from helper import draw_sample
from ujipen.ujipen_class import UJIPen


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
            plt.suptitle(f"{k_neighbors} closest patterns to the input sample (left)")
            plt.show()


def test_dtw(fold="train", k_neighbors=5):
    ujipen = UJIPen()
    samples = ujipen.get_samples(fold=fold)
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
    print(f"{fold} accuracy: {accuracy}")
    cm = confusion_matrix(labels_true, labels_predicted)
    plt.imshow(cm)
    plt.show()


if __name__ == '__main__':
    # show_knn(k_neighbors=5)
    test_dtw(fold='test', k_neighbors=1)

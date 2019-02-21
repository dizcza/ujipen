from dtw import dtw
import math
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm

from constants import POINTS_DIR, CHAR_PATTERNS_H, CHAR_PATTERNS_C, WIDTH, HEIGHT, PATTERN_SIZE, ASPECT_RATIO_MAX
from dtw_solver import l2_dist, dtw_vanilla, dtw_online, dtw_vectorized

SCREEN_CENTER = HEIGHT / 2, WIDTH / 2

H_FILE_HEADER = f"""/*
 * char_patterns.h
 *
 *  Created on: Jan 31, 2019
 *      Author: Danylo Ulianych
 */

#ifndef CHAR_PATTERNS_H_
#define CHAR_PATTERNS_H_

#define ALPHABET_SIZE     {len(list(POINTS_DIR.iterdir()))}
#define PATTERN_SIZE      {PATTERN_SIZE}
#define ASPECT_RATIO_MAX  {ASPECT_RATIO_MAX}

#include <stdint.h>
#include "arm_math.h"
"""


def load_points():
    points = {}
    for filepath in sorted(POINTS_DIR.iterdir()):
        points[filepath.stem] = np.loadtxt(filepath, dtype=np.int32, delimiter=',')
    return points


def reduce(points):
    for letter_key, letter_points in points.items():
        if len(letter_points) < PATTERN_SIZE:
            warnings.warn(f"Char {letter_key} has {len(letter_points)} < {PATTERN_SIZE} points.")
        indices = np.linspace(0, len(letter_points) - 1, num=PATTERN_SIZE, endpoint=True, dtype=np.int32)
        letter_points = letter_points[indices]
        points[letter_key] = letter_points
    return points


def normalize(points):
    for letter_key, letter_points in points.items():
        y, x = letter_points.T
        ymin, xmin, ymax, xmax = y.min(), x.min(), y.max(), x.max()
        h = ymax - ymin
        w = xmax - xmin
        if w / h > ASPECT_RATIO_MAX:
            # height is too small
            h = w
        elif h / w > ASPECT_RATIO_MAX:
            # width is too small
            w = h
        yc = (ymin + ymax) / 2
        xc = (xmin + xmax) / 2
        y = 0.5 + (y - yc) / h
        x = 0.5 + (x - xc) / w
        points[letter_key] = np.column_stack((y, x))
    return points


def display(points, margin=0.02):
    n = len(points)
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    for i, letter_key in enumerate(points.keys(), start=1):
        plt.subplot(rows, cols, i)
        letter_points = points[letter_key].copy()
        dv = letter_points[1:] - letter_points[:-1]
        dist = np.linalg.norm(dv, axis=1)
        corners = np.where(dist > 0.2)[0]
        corners += 1
        letter_points[:, 0] = letter_points[:, 0].max() - letter_points[:, 0]
        for chunk in np.split(letter_points, corners):
            y, x = chunk.T
            plt.plot(x, y)
        plt.xlim(left=0 - margin, right=1 + margin)
        plt.ylim(bottom=0 - margin, top=1 + margin)
        plt.axis('off')
        plt.title(letter_key)
    plt.show()


def print_info(points):
    for letter_key, letter_points in points.items():
        y, x = letter_points.T
        print(f"{letter_key}: ymin={x.min()}, ymax={y.max()}, xmin={x.min()}, xmax={x.max()}")


def convert_to_c(dtype='float32_t', separate=True):
    points = load_points()
    points = reduce(points)
    points = normalize(points)
    print_info(points)
    display(points)
    assert set(map(len, points.values())) == {PATTERN_SIZE}
    alphabet = sorted(points.keys())
    pattern_coords_decl = lambda suffix: f"const {dtype} PATTERN_COORDS_{suffix}[ALPHABET_SIZE][PATTERN_SIZE]"

    h_lines = [
        H_FILE_HEADER,
        "\nextern const uint8_t ALPHABET[ALPHABET_SIZE];\n",
        f"\nextern {pattern_coords_decl('X')};",
        f"\nextern {pattern_coords_decl('Y')};",
        "\n\n#endif /* CHAR_PATTERNS_H_ */"
    ]
    with open(CHAR_PATTERNS_H, 'w') as f:
        f.write(''.join(h_lines))

    c_lines = [
        "#include \"char_patterns.h\"",
        "\n\nconst uint8_t ALPHABET[ALPHABET_SIZE] = \"{alphabet}\";\n".format(alphabet=''.join(alphabet)),
    ]
    if separate:
        def write_single_coordinates(one_dim_coords: list, suffix: str):
            c_lines.append(f"\n{pattern_coords_decl(suffix)} = {{")
            for x in one_dim_coords:
                x = [f"{xval:.3f}f" for xval in x]
                c_lines.append('\n\t{')
                c_lines.append(', '.join(x))
                c_lines.append('},')
            c_lines.append('\n};')

        ys, xs = [], []
        for letter_key in alphabet:
            y, x = points[letter_key].T
            ys.append(y)
            xs.append(x)
        write_single_coordinates(xs, suffix='X')
        write_single_coordinates(ys, suffix='Y')
    else:
        c_lines.extend(["\n/* ",
                        "\n * For each symbol in the ALPHABET,",
                        "\n * specify PATTERN_SIZE pairs of {y, x} coordinates.",
                        "\n */"])
        c_lines.append(f"\nconst {dtype} PATTERN_COORDS[ALPHABET_SIZE][PATTERN_SIZE][2] = {{")
        for letter_key in alphabet:
            c_lines.append('\n\t{')
            for p in points[letter_key]:
                c_lines.append("{{{y}, {x}}}, ".format(y=p[0], x=p[1]))
            c_lines.append('},')
        c_lines.append('\n};')
    with open(CHAR_PATTERNS_C, 'w') as f:
        f.write(''.join(c_lines))

    print(f"Saved to {CHAR_PATTERNS_H.parent}")


def dtw_test():
    np.random.seed(26)
    points = {}
    for filepath in POINTS_DIR.iterdir():
        points[filepath.stem] = np.loadtxt(filepath, dtype=np.int32, delimiter=',')
    for key, letter in points.items():
        print(f"{key}: {len(letter)}")
    keys = list(points.keys())
    for letter_id, letter1_key in enumerate(keys[:-1]):
        for letter2_key in keys[letter_id + 1:]:
            letter1 = points[letter1_key]
            letter2 = points[letter2_key]
            d, cost_matrix, acc_cost_matrix, path = dtw(letter1, letter2, dist=l2_dist)
            dist_online = dtw_online(sample=letter1, pattern=letter2)
            assert np.isclose(d, dist_online)
            print(f"d('{letter1_key}', '{letter2_key}') = {d}")
    return points


def my_dtw_test():
    for trial in tqdm(range(10)):
        sample = np.random.randn(100, 2)
        pattern = np.random.randn(100, 2)
        dist_online = dtw_online(sample, pattern)
        # dist_matrix = dtw_vanilla(sample, pattern)
        dist_matrix_vectorized = dtw_vectorized(sample, pattern)
        dist_ref, cost_matrix, acc_cost_matrix, path = dtw(sample, pattern, dist=l2_dist)
        assert np.allclose(dist_matrix_vectorized, acc_cost_matrix)
        # assert np.isclose(dist_online, dist_matrix[-1, -1])


if __name__ == '__main__':
    my_dtw_test()
    # dtw_test()

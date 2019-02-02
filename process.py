from dtw import dtw
import numpy as np
import warnings

from constants import POINTS_DIR, CHAR_PATTERNS_H, CHAR_PATTERNS_C, WIDTH, HEIGHT, PATTERN_SIZE

SCREEN_CENTER = HEIGHT / 2, WIDTH / 2

dist_func = lambda x, y: sum((x - y) ** 2)

H_FILE_HEADER = f"""/*
 * char_patterns.h
 *
 *  Created on: Jan 31, 2019
 *      Author: Danylo Ulianych
 */

#ifndef CHAR_PATTERNS_H_
#define CHAR_PATTERNS_H_

#define ALPHABET_SIZE {len(list(POINTS_DIR.iterdir()))}
#define PATTERN_SIZE  {PATTERN_SIZE}

#include <stdint.h>
"""


def load_points():
    points = {}
    for filepath in sorted(POINTS_DIR.iterdir()):
        points[filepath.stem] = np.loadtxt(filepath, dtype=np.int32, delimiter=',')
    return points


def align(points):
    for letter_key, letter_points in points.items():
        y, x = letter_points.T
        ymin, xmin, ymax, xmax = y.min(), x.min(), y.max(), x.max()
        box_center = (ymin + ymax) / 2, (xmin + xmax) / 2
        v_to_center = np.subtract(SCREEN_CENTER, box_center)
        v_to_center = v_to_center.astype(letter_points.dtype)
        letter_points = letter_points + v_to_center
        points[letter_key] = letter_points
    return points


def reduce(points):
    for letter_key, letter_points in points.items():
        if len(letter_points) < PATTERN_SIZE:
            warnings.warn(f"Char {letter_key} has {len(letter_points)} < {PATTERN_SIZE} points.")
        indices = np.linspace(0, len(letter_points) - 1, num=PATTERN_SIZE, endpoint=True, dtype=np.int32)
        letter_points = letter_points[indices]
        points[letter_key] = letter_points
    return points


def scale_to(points, height, width):
    for letter_key, letter_points in points.items():
        letter_points[:, 0] = letter_points[:, 0] / HEIGHT * height
        letter_points[:, 1] = letter_points[:, 1] / WIDTH * width
        points[letter_key] = letter_points
    return points


def convert_to_c(dtype='uint16_t', separate=True):
    points = load_points()
    points = align(points)
    points = scale_to(points, height=320, width=240)
    points = reduce(points)
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
                x = x.astype(str)
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
            d, cost_matrix, acc_cost_matrix, path = dtw(letter1, letter2, dist=dist_func)
            d2 = dtw(letter2, letter1, dist=dist_func)[0]
            assert np.isclose(d, d2)
            print(f"d('{letter1_key}', '{letter2_key}') = {d}")
    return points


if __name__ == '__main__':
    convert_to_c()

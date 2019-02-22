import numpy as np
import string

from constants import *
from ujipen.clustering import UJIPen
from manual.loader import load_manual_patterns

H_FILE_HEADER = f"""/*
 * char_patterns.h
 *
 *  Created on: Jan 31, 2019
 *      Author: Danylo Ulianych
 */

#ifndef CHAR_PATTERNS_H_
#define CHAR_PATTERNS_H_

#define PATTERN_SIZE      {PATTERN_SIZE}
"""


def make_patterns_fixed_size(patterns):
    patterns_fixed_size = {}
    for word in patterns.keys():
        patterns_fixed_size[word] = []
        for trial in patterns[word]:
            indices = np.linspace(0, len(trial) - 1, num=PATTERN_SIZE, endpoint=True, dtype=np.int32)
            trial = trial[indices]
            patterns_fixed_size[word].append(trial)
    return patterns_fixed_size


def convert_to_c(patterns, dtype='float32_t'):
    patterns = make_patterns_fixed_size(patterns)
    total_patterns = sum(map(len, patterns.values()))
    assert ''.join(sorted(patterns.keys())) == string.ascii_lowercase
    for word in patterns.keys():
        assert set(map(len, patterns[word])) == {PATTERN_SIZE}
    pattern_coords_decl = lambda suffix: f"const {dtype} PATTERN_COORDS_{suffix}[TOTAL_PATTERNS][PATTERN_SIZE]"

    h_lines = [
        H_FILE_HEADER,
        f"#define TOTAL_PATTERNS    {total_patterns}\n"
        "\n#include <stdint.h>"
        "\n#include \"arm_math.h\"\n"
        "\nextern const uint8_t PATTERN_LABEL[TOTAL_PATTERNS];\n",
        f"\nextern {pattern_coords_decl('X')};",
        f"\nextern {pattern_coords_decl('Y')};",
        "\n\n#endif /* CHAR_PATTERNS_H_ */"
    ]
    with open(CHAR_PATTERNS_H, 'w') as f:
        f.write(''.join(h_lines))

    labels_true = []
    for word in patterns.keys():
        labels_true.extend([word] * len(patterns[word]))
    labels_true = ''.join(labels_true)
    c_lines = [
        "#include \"char_patterns.h\"",
        f"\n\nconst uint8_t PATTERN_LABEL[TOTAL_PATTERNS] = \"{labels_true}\";\n",
    ]
    
    def write_single_coordinates(one_dim_coords: list, suffix: str):
        c_lines.append(f"\n{pattern_coords_decl(suffix)} = {{")
        for x in one_dim_coords:
            x = [f"{xval:.3f}f" for xval in x]
            c_lines.append('\n\t{')
            c_lines.append(', '.join(x))
            c_lines.append('},')
        c_lines.append('\n};')

    xs, ys = [], []
    for word in patterns.keys():
        for sample in patterns[word]:
            x, y = sample.T
            xs.append(x)
            ys.append(y)
    write_single_coordinates(xs, suffix='X')
    write_single_coordinates(ys, suffix='Y')
    with open(CHAR_PATTERNS_C, 'w') as f:
        f.write(''.join(c_lines))

    print(f"Saved to {CHAR_PATTERNS_H.parent}")


if __name__ == '__main__':
    patterns = UJIPen().get_min_intra_dist_patterns()
    # patterns = load_manual_patterns()
    convert_to_c(patterns)

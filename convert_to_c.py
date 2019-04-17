import string
import numpy as np
from typing import List

from constants import *
from preprocess import equally_spaced_points_patterns, normalize_patterns_fixed_point
from ujipen.ujipen_class import UJIPen
from ujipen.clustering import ujipen_cluster
from helper import check_unique_patterns
from manual.loader import load_manual_patterns


def convert_to_c(patterns, q7_t=False):
    patterns = equally_spaced_points_patterns(patterns)
    if q7_t:
        patterns = normalize_patterns_fixed_point(patterns)
    total_patterns = sum(map(len, patterns.values()))
    assert check_unique_patterns(patterns)
    pattern_coords_decl = lambda suffix: f"const float_coord PATTERN_COORDS_{suffix}[TOTAL_PATTERNS][PATTERN_SIZE]"

    define_q7_t = "#define CHAR_PATTERNS_DATATYPE_Q7"
    if not q7_t:
        define_q7_t = '//' + define_q7_t

    h_header = f"""
/*
 * char_patterns.h
 *
 *  Created on: Jan 31, 2019
 *      Author: Danylo Ulianych
 */

#ifndef CHAR_PATTERNS_H_
#define CHAR_PATTERNS_H_

// do not modify this
{define_q7_t}

#define PATTERN_SIZE      {PATTERN_SIZE}
#define TOTAL_PATTERNS    {total_patterns}

#include <stdint.h>
#include "arm_math.h"

#ifdef CHAR_PATTERNS_DATATYPE_Q7
#define CHAR_PATTERNS_RESOLUTION  0.0078125f

typedef q7_t float_coord;
#else
#define CHAR_PATTERNS_RESOLUTION  0.0f

typedef float32_t float_coord;
#endif  /* CHAR_PATTERNS_DATATYPE_Q7 */

typedef struct CharPattern {{
    float_coord *xcoords, *ycoords;
    uint32_t size;
}} CharPattern;

typedef struct CharPattern_PredictedInfo {{
    char predicted_char;
    uint32_t duration;
    float32_t distance;
}} CharPattern_PredictedInfo;

extern const uint8_t PATTERN_LABEL[TOTAL_PATTERNS];
extern {pattern_coords_decl('X')};
extern {pattern_coords_decl('Y')};

#endif /* CHAR_PATTERNS_H_ */
"""

    with open(CHAR_PATTERNS_H, 'w') as f:
        f.write(h_header)

    labels_true = []
    for word in patterns.keys():
        labels_true.extend([word] * len(patterns[word]))
    labels_true = ''.join(labels_true)
    c_lines = [
        "#include \"char_patterns.h\"",
        f"\n\nconst uint8_t PATTERN_LABEL[TOTAL_PATTERNS] = \"{labels_true}\";\n",
    ]

    def write_single_coordinates(one_dim_coords: List[np.ndarray], suffix: str):
        c_lines.append(f"\n{pattern_coords_decl(suffix)} = {{")
        for x in one_dim_coords:
            if q7_t:
                x = x & 0xff
                x = [f"0x{xval:02X}" for xval in x]
            else:
                x = [f"{xval:.3f}f" for xval in x]
            c_lines.append('\n\t{')
            c_lines.append(', '.join(x))
            c_lines.append('},')
        c_lines.append('\n};')

    xs, ys = [], []
    for word in patterns.keys():
        for sample in patterns[word]:
            x, y = np.vstack(sample).T
            xs.append(x)
            ys.append(y)
    write_single_coordinates(xs, suffix='X')
    write_single_coordinates(ys, suffix='Y')
    with open(CHAR_PATTERNS_C, 'w') as f:
        f.write(''.join(c_lines))

    print(f"Saved to {CHAR_PATTERNS_H.parent}")


if __name__ == '__main__':
    ujipen_cluster()
    patterns = UJIPen().get_min_intra_dist_patterns()
    # patterns = load_manual_patterns()
    convert_to_c(patterns, q7_t=True)

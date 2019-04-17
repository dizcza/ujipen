from pathlib import Path

PATTERN_SIZE = 30
VERT_SLANT_ANGLE = 50

CHAR_PATTERNS_H = Path.home().joinpath("microcontrollers/workspace/f429-chars/TrueSTUDIO/f429-chars/DTW/char_patterns.h")
CHAR_PATTERNS_C = CHAR_PATTERNS_H.with_suffix('.c')

MODELS_DIR = Path(__file__).with_name('models')

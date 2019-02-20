from pathlib import Path

WIDTH = 200
HEIGHT = 200

PATTERN_SIZE = 50
ASPECT_RATIO_MAX = 4

POINTS_DIR = Path("points")
IMAGES_DIR = Path("images")
CHAR_PATTERNS_H = Path("/home/dizcza/microcontrollers/STM32_Projects/f429-chars/TrueSTUDIO/f429-chars/DTW/char_patterns.h")
CHAR_PATTERNS_C = CHAR_PATTERNS_H.with_suffix('.c')

POINTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

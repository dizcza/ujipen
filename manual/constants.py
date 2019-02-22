from pathlib import Path

WIDTH = 200
HEIGHT = 200

PATTERN_SIZE = 50
ASPECT_RATIO_MAX = 4

MANUAL_DIR = Path(__file__).parent

POINTS_DIR = MANUAL_DIR / "points"
IMAGES_DIR = MANUAL_DIR / "images"

POINTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

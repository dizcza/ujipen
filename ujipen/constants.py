from pathlib import Path

UJIPEN_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/uji-penchars/version2/ujipenchars2.txt"
UJIPEN_DIR = Path(__file__).parent.parent / "ujipenchars2"
UJIPEN_TXT = UJIPEN_DIR / "ujipenchars2.txt"
UJIPEN_PKL = UJIPEN_TXT.with_suffix('.pkl')
UJIPEN_INTRA_DIST_PATH = UJIPEN_DIR / "intra-dist.pkl"

TRIALS_KEY = "trials"  # samples
INTRA_DIST_KEY = "intra-dist"  # intra-class dist matrix
LABELS_KEY = "clusters"  # sample cluster ID

VERT_SLANGE_ANGLE = 50

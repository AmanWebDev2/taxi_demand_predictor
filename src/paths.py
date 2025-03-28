from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = PARENT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"

if not Path(DATA_DIR).exists():
    os.makedirs(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.makedirs(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.makedirs(TRANSFORMED_DATA_DIR)

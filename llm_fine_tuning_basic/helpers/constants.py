"""Useful constants, especially paths."""

from pathlib import Path

MODULE_PATH = Path(__file__)
HELPERS_PATH = MODULE_PATH.parent
DIR = HELPERS_PATH.parent
DATA_DIR = DIR.joinpath("data")
LABELS_FILE_PATH = DIR.joinpath("aux_data").joinpath("labels.json")

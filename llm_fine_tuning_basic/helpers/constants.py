"""Useful constants, especially paths."""

from pathlib import Path

MODULE_PATH = Path(__file__)
HELPERS_PATH = MODULE_PATH.parent
DIR = HELPERS_PATH.parent
DATA_DIR = DIR.joinpath("data")
LABELS_FILE_PATH = DIR.joinpath("aux_data").joinpath("labels.json")
RUNS_DIRECTORY = DIR.joinpath("runs")
MANIFEST_FILE_NAME = "manifest.json"
EVALUATION_RESULTS_FILE_NAME = "test.jsonl"
PROMPT_DIR = DIR.joinpath("prompts")
CLASSIFICATION_REPORT_FILE_NAME = "classification_report.json"
TRAINING_DATA_DIR = DIR.joinpath("training_data")

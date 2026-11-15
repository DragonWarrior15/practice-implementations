"""Useful constants, especially paths."""

from pathlib import Path

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-bf16"
MODULE_PATH = Path(__file__)
HELPERS_PATH = MODULE_PATH.parent
DIR = HELPERS_PATH.parent
DATA_DIR = DIR.joinpath("data")
RUNS_DIRECTORY = DIR.joinpath("runs")
MANIFEST_FILE_NAME = "manifest.json"
EVALUATION_RESULTS_FILE_NAME = "test.jsonl"
TRAINING_DATA_DIR = DIR.joinpath("training_data")

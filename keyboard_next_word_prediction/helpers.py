"""Helper functions for the project."""

from datasets import load_dataset

DATASET_NAME = "wikitext-103-raw-v1"
DATA_DIR = "data"


def download_data():
    ds = load_dataset("Salesforce/wikitext", DATASET_NAME)
    ds.save_to_disk(DATA_DIR)


def load_data():
    ds = load_dataset(DATA_DIR)
    return ds

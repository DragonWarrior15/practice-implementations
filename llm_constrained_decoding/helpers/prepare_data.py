"""Helper script to download the dataset BANKING77."""

import argparse
import json

from datasets import load_dataset, load_from_disk
from helpers.constants import (
    DATA_DIR,
    LABELS_FILE_PATH,
)


def download_data():
    """Downloads the banking77 dataset and saves to data dir."""
    dataset = load_dataset("PolyAI/banking77")
    dataset.save_to_disk(DATA_DIR)


def get_and_save_labels():
    """Extract labels for reuse in later pipelines."""
    dataset = load_from_disk(DATA_DIR)
    train = dataset["train"]
    label_names = train.features["label"].names

    LABELS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    LABELS_FILE_PATH.write_text(json.dumps(label_names))


def main():
    # parse args
    parser = argparse.ArgumentParser(description="Dataset handling utility.")
    parser.add_argument(
        "--step",
        type=str,
        help="Step to perform",
        choices=(
            "download",
            "nothing",
            "save_labels",
        ),
        default="nothing",
    )

    args = parser.parse_args()
    step = args.step

    if step == "nothing":
        return
    elif step == "download":
        download_data()
    elif step == "save_labels":
        get_and_save_labels()
    else:
        raise ValueError("Unrecognized argument for step")


if __name__ == "__main__":
    main()

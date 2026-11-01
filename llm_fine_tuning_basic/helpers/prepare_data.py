"""Helper script to download the dataset BANKING77."""

import argparse

from datasets import load_dataset, load_from_disk
from helpers.constants import DATA_DIR


def download_data():
    """Downloads the banking77 dataset and saves to data dir."""
    dataset = load_dataset("PolyAI/banking77")
    dataset.save_to_disk(DATA_DIR)


def get_dataset_stats():
    """Print the basic stats of the data."""
    dataset = load_from_disk(DATA_DIR)

    print(dataset)  # get splits, columns, and row counts

    # look at the splits
    train = dataset["train"]
    test = dataset["test"]

    print("Total training data rows: ", len(train))
    print(train.column_names)
    print(train[0])  # First example
    print(train[:3])  # First three examples, grouped by column
    print(train[0]["text"])

    print("Total test data rows: ", len(test))


def main():
    # parse args
    parser = argparse.ArgumentParser(description="Dataset handling utility.")
    parser.add_argument(
        "--step",
        type=str,
        help="Step to perform",
        choices=("download", "prepare_json", "nothing", "get_dataset_stats"),
        default="nothing",
    )

    args = parser.parse_args()
    step = args.step

    if step == "nothing":
        return
    elif step == "prepare_json":
        return
    elif step == "download":
        download_data()
    elif step == "get_dataset_stats":
        get_dataset_stats()
    else:
        raise ValueError("Unrecognized argument for step")


if __name__ == "__main__":
    main()

"""Helper script to download the dataset BANKING77."""

import argparse

from datasets import load_dataset, load_from_disk
from helpers.constants import DATA_DIR


def download_data():
    """Downloads the banking77 dataset and saves to data dir."""
    dataset = load_dataset("trl-lib/ultrafeedback_binarized")
    dataset.save_to_disk(DATA_DIR)


def get_dataset_stats():
    """Print the basic stats of the data."""
    dataset = load_from_disk(DATA_DIR)

    print(dataset)  # get splits, columns, and row counts

    # look at the splits
    train = dataset["train"]
    test = dataset["test"]

    print("Total training data rows: ", len(train))
    
    print("Training Data Column Names")
    print(train.column_names)

    train_0 = train[0]
    
    print("Training Data Example 0:")
    print("Length")
    print(len(train_0["chosen"]))
    
    print("These are messages, printing roles and first 10 chars of content")
    for idx, item in enumerate(train_0["chosen"]):
        print(idx, item["role"], item["content"][:20])

    print("Training Data Example 0: Scores")
    print("score_chosen: ", train_0["score_chosen"])
    print("score_rejected: ", train_0["score_rejected"])

    print("Total test data rows: ", len(test))


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
            "get_dataset_stats",
        ),
        default="nothing",
    )

    args = parser.parse_args()
    step = args.step

    if step == "nothing":
        return
    elif step == "download":
        download_data()
    elif step == "get_dataset_stats":
        get_dataset_stats()
    else:
        raise ValueError("Unrecognized argument for step")


if __name__ == "__main__":
    main()

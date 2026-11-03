"""Helper script to download the dataset BANKING77."""

import argparse
import json
from datetime import UTC, datetime

from datasets import load_dataset, load_from_disk
from helpers.constants import (
    DATA_DIR,
    LABELS_FILE_PATH,
    MANIFEST_FILE_NAME,
    TRAINING_DATA_DIR,
)
from helpers.utils import get_system_prompt, read_labels


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

    # print total labels
    label_names = train.features["label"].names
    print("Sample Labels")
    print(label_names[:10])


def convert_to_json(system_prompt: str, text: str, label: str):
    """Simple utility that takes in a prompt, along with a text and label
    and generates the json format for input to a language model.
    """
    json_template = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            # note that we are also preparing the label in the same message
            # flow, this will not work in practice as we will be leaking the
            # output into the model which is not ideal
            {"role": "assistant", "content": label},
        ]
    }

    return json_template


def prepare_json(
    system_prompt: str | None = None,
) -> list[dict[str, list[dict[str, str]]]]:
    """Prepares jsonl data and saves to respective files."""
    # for the basic version, read the training data, select one example
    # and generate the json
    dataset = load_from_disk(DATA_DIR)
    train = dataset["train"]

    example = train[0]

    label_names = train.features["label"].names

    if not system_prompt:
        system_prompt = (
            "Classify the banking request into exactly one of these intents:\n"
            + "\n".join(label_names)
            + "\nReturn only the exact intent name, with no explanation."
        )

    print(
        convert_to_json(
            system_prompt=system_prompt,
            text=example["text"],
            label=label_names[example["label"]],
        )
    )


def prepare_finetuning_data(
    prompt_file_name: str = "prompt_v1.md",
    valid_split: float = 0.2,
    seed: int = 42,
):
    """Prepares the finetuning data and saves it"""
    # make the dir so it exists
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # same some manifest information
    manifest = {
        "valid_split": valid_split,
        "seed": seed,
        "prompt_file": prompt_file_name,
        "created_at": datetime.now(UTC).isoformat(),
    }

    TRAINING_DATA_DIR.joinpath(MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, indent=2)
    )

    # get the system prompt and labels
    system_prompt = get_system_prompt(prompt_file_name)
    labels = read_labels()

    # read the train data, and split to train and test sets
    dataset = load_from_disk(DATA_DIR)

    # look at the splits
    train = dataset["train"]
    test = dataset["test"]

    # shuffle and then split
    train = train.shuffle(seed=seed)
    train_indices = int(len(train) * (1 - valid_split))
    train, valid = (
        train.select(range(train_indices)),
        train.select(range(train_indices, len(train))),
    )

    # iterate over each dataset, and build the datasets for finetuning
    # in the required format, json with json entries as prepare_json
    for curr_dataset, dataset_name in zip(
        [train, valid, test], ["train", "valid", "test"]
    ):
        with open(
            TRAINING_DATA_DIR.joinpath(dataset_name + ".jsonl"),
            "w",
            encoding="utf-8",
        ) as f:
            for row in curr_dataset:
                curr_json = convert_to_json(
                    system_prompt=system_prompt,
                    text=row["text"],
                    label=labels[row["label"]],
                )
                f.write(json.dumps(curr_json) + "\n")


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
            "prepare_json",
            "nothing",
            "get_dataset_stats",
            "save_labels",
            "prepare_finetuning_data",
        ),
        default="nothing",
    )

    args = parser.parse_args()
    step = args.step

    if step == "nothing":
        return
    elif step == "prepare_json":
        prepare_json()
    elif step == "download":
        download_data()
    elif step == "get_dataset_stats":
        get_dataset_stats()
    elif step == "save_labels":
        get_and_save_labels()
    elif step == "prepare_finetuning_data":
        prepare_finetuning_data()
    else:
        raise ValueError("Unrecognized argument for step")


if __name__ == "__main__":
    main()

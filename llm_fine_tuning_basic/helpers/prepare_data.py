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
        prepare_json()
    elif step == "download":
        download_data()
    elif step == "get_dataset_stats":
        get_dataset_stats()
    else:
        raise ValueError("Unrecognized argument for step")


if __name__ == "__main__":
    main()

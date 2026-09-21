"""Helper functions for the project."""

import json
from collections import defaultdict

from datasets import DatasetDict, load_dataset, load_from_disk

# DATASET_NAME = "wikitext-103-raw-v1"
# DATA_DIR = "data"

DATASET_NAME = "wikitext-2-raw-v1"
DATA_DIR = "data_wikitext2"


def download_data():
    ds = load_dataset("Salesforce/wikitext", DATASET_NAME)
    ds.save_to_disk(DATA_DIR)


def load_data() -> DatasetDict:
    ds = load_from_disk(DATA_DIR)
    return ds


def process_sentence(sent: str) -> list[str]:
    """Apply standard preprocessing on dataset and yield a list of sentences."""
    sent_processed = sent.strip().split()
    return sent_processed


def get_upto_trigram_from_list(sent: list | list[list]) -> dict[int, list]:
    """Return unigrams, bigrams and trigrams from a list."""
    output = {idx: defaultdict(int) for idx in range(1, 4)}

    if not sent:
        return output

    if not isinstance(sent[0], list):
        sent = [sent]

    for step_count in range(1, 4):
        for sen in sent:
            for start_idx in range(len(sen) - step_count + 1):
                if len(sen[start_idx : start_idx + step_count]) == step_count:
                    output[step_count][
                        tuple(sen[start_idx : start_idx + step_count])
                    ] += 1

    return output


def save_ngram_model(model: dict[tuple, int], model_path: str):
    """Save a dict as JSON. Converts tuple to space separated list."""
    model_serialized = {" ".join(k): v for k, v in model.items()}

    with open(model_path, "w") as f:
        json.dump(model_serialized, f, indent=2)

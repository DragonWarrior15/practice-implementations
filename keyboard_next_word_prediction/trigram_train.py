"""Module for training and testing trigram models."""

import json
import os

from helpers import (
    get_upto_trigram_from_list,
    load_data,
    process_sentence,
    save_ngram_model,
)

MODEL_PATH = "artifacts/trigram_model"
TRIGRAM_MODEL_PATH = os.path.join(MODEL_PATH, "trigram_dictionary.json")
BIGRAM_MODEL_PATH = os.path.join(MODEL_PATH, "bigram_dictionary.json")
UNIGRAM_MODEL_PATH = os.path.join(MODEL_PATH, "unigram_dictionary.json")


def build_trigram_dictionary():
    """Load training data, and build the trigram dictionary."""

    # load the training data
    ds = load_data()["train"]

    # base global counters
    models = get_upto_trigram_from_list([])

    sent_processed = None

    for ds_idx in range(len(ds)):
        # check if new line, new lines serve as a breaking point to reset
        # prev_sent_processed
        # prev_sent_processed is needed to calculate ngrams across sentences
        if not ds[ds_idx]["text"].strip():
            continue

        sent_processed = process_sentence(ds[ds_idx]["text"])

        models_curr = get_upto_trigram_from_list(sent_processed)

        for k in models_curr:
            for k1 in models_curr[k]:
                models[k][k1] += models_curr[k][k1]

    if not models:
        return

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH, exist_ok=True)

    # save the models
    for step_count, model_path in [
        [1, UNIGRAM_MODEL_PATH],
        [2, BIGRAM_MODEL_PATH],
        [3, TRIGRAM_MODEL_PATH],
    ]:
        # save
        save_ngram_model(models[step_count], model_path)

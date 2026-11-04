"""Helper script to run end to end evaluations."""

# some objects to define data exchanges across different functions
import argparse
import json
import logging
from datetime import UTC, datetime

from datasets import Dataset, load_from_disk
from helpers.constants import (
    CLASSIFICATION_REPORT_FILE_NAME,
    DATA_DIR,
    EVALUATION_RESULTS_FILE_NAME,
    MANIFEST_FILE_NAME,
    RUNS_DIRECTORY,
)
from helpers.utils import (
    get_model_and_tokenizer,
    get_response_from_llm,
    read_labels,
)
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Manifest stores information about how to run a model
class Manifest(BaseModel):
    run_id: str
    prompt_file: str
    model: str
    started_at: str
    completed_at: str | None = None
    notes: str = ""
    evaluation_type: str = "test"
    adapter_path: str | None = None


class EvaluationResult(BaseModel):
    id: int
    input: str
    actual: str
    predicted: str


def build_and_save_manifest(
    prompt_file: str, model: str, evaluation_type: str, adapter_path: str
) -> Manifest:
    """Build and saves the manifest."""
    # fix the time
    now_dt = datetime.now(UTC)

    manifest = Manifest(
        run_id=now_dt.strftime(r"%Y-%m-%dT%H%M%SZ"),
        prompt_file=prompt_file,
        model=model,
        started_at=now_dt.isoformat(),
        evaluation_type=evaluation_type,
        adapter_path=adapter_path,
    )

    run_directory = RUNS_DIRECTORY.joinpath(manifest.run_id)
    run_directory.mkdir(parents=True, exist_ok=False)
    run_directory.joinpath(MANIFEST_FILE_NAME).write_text(
        manifest.model_dump_json(indent=2)
    )

    # return the manifest for downstream consumption
    return manifest


def get_manifest_from_run_id(run_id: str) -> Manifest | None:
    """Based on the run id, explore the runs folder to read a run manifest."""
    manifest_file_path = RUNS_DIRECTORY.joinpath(run_id).joinpath(
        MANIFEST_FILE_NAME
    )
    if manifest_file_path.exists():
        try:
            # read the file
            manifest = Manifest.model_validate_json(
                manifest_file_path.read_text()
            )
            return manifest
        except Exception as e:
            logger.error(
                f"Could not build manifest for run id: {run_id} due to Exception {str(e)}"
            )
            return None

    # if the directory is not found
    return None


def run_evaluation_for_manifest(manifest: Manifest, dataset: Dataset) -> None:
    """Run inference and persist each case result incrementally."""
    # for each case in the evaluation dataset, generate the outputs and build
    # the EvaluationResult instance
    run_directory = RUNS_DIRECTORY.joinpath(manifest.run_id)
    results_path = run_directory.joinpath(EVALUATION_RESULTS_FILE_NAME)

    model, tokenizer = get_model_and_tokenizer(
        manifest.model, adapter_path=manifest.adapter_path
    )

    # get the labels list
    labels = read_labels()

    with results_path.open("a") as results_file:
        for idx, row in enumerate(dataset):
            text = row["text"]
            label = row["label"]

            llm_response = get_response_from_llm(
                user_query=text,
                prompt_file_name=manifest.prompt_file,
                model=model,
                tokenizer=tokenizer,
            )

            case_result = EvaluationResult(
                id=idx,
                input=text,
                actual=labels[label],
                predicted=llm_response,
            )
            # Persist each result immediately so interrupted runs retain their
            # completed cases. Each line is an independent JSON document.
            results_file.write(case_result.model_dump_json() + "\n")
            results_file.flush()

    manifest.completed_at = datetime.now(UTC).isoformat()

    run_directory = RUNS_DIRECTORY.joinpath(manifest.run_id)
    run_directory.joinpath(MANIFEST_FILE_NAME).write_text(
        manifest.model_dump_json(indent=2)
    )


def build_and_save_classification_report(
    manifest: Manifest | None = None, run_id: str | None = None
):
    """Build a summary of different metrics when running evaluation."""
    # first get the relevant data
    if not manifest and not run_id:
        logger.warning(
            "Skipping build_and_save_classification_report as neither"
            " manifest, nor run_id are supplied"
        )

    # if run id is given, build the manifest object
    if not manifest and run_id:
        manifest = get_manifest_from_run_id(run_id)

    # manifest is available, read the predictions to get the predictions and actuals
    run_dir = RUNS_DIRECTORY.joinpath(manifest.run_id)
    results_path = run_dir.joinpath(EVALUATION_RESULTS_FILE_NAME)

    # get the labels for input to the classification report
    labels = read_labels()

    y_true = []
    y_predicted = []
    labels_set = set(labels)
    invalid_labels_count = 0

    with open(results_path, encoding="utf-8") as results_file:
        for line in results_file:
            result = EvaluationResult.model_validate_json(line)
            y_true.append(result.actual)
            y_predicted.append(result.predicted)
            if result.predicted not in labels_set:
                invalid_labels_count += 1

    report = classification_report(
        y_true=y_true, y_pred=y_predicted, labels=labels, output_dict=True
    )

    # add an invalid label rate as well
    invalid_label_rate = invalid_labels_count / len(y_true)
    report["invalid_label_rate"] = invalid_label_rate

    # add an explicit accuracy metric since there are invalid labels
    report["accuracy"] = accuracy_score(y_true, y_predicted)

    # save the report
    run_dir.joinpath(CLASSIFICATION_REPORT_FILE_NAME).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def main():
    # parse args
    parser = argparse.ArgumentParser(description="Evaluation Runner Utility.")

    # the file name for the prompt
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="Name of the Prompt File.",
        choices=["prompt_v1.md"],
        default="prompt_v1.md",
        required=False,
    )

    # name of the model
    parser.add_argument(
        "--model",
        type=str,
        help="Full name of the model.",
        choices=["mlx-community/Llama-3.2-1B-Instruct-bf16"],
        default="mlx-community/Llama-3.2-1B-Instruct-bf16",
        required=False,
    )

    # run id if running just evaluations
    parser.add_argument(
        "--run_id",
        type=str,
        help="Run ID to evaluate.",
        required=False,
    )

    # evaluation type, valid or test
    parser.add_argument(
        "--evaluation_type",
        type=str,
        help="Evaluation Type, valid or test",
        choices=["valid", "test"],
        default="test",
        required=False,
    )

    # adapter path for fine tuned models
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Adapter directory, containing the adapter config and safetensors",
        default=None,
        required=False,
    )

    args = parser.parse_args()

    if args.run_id:
        build_and_save_classification_report(run_id=args.run_id)
        return

    # read the train data, and split to train and test sets
    dataset = load_from_disk(DATA_DIR)

    # check the dataset to pick
    if args.evaluation_type == "test":
        dataset = dataset["test"]
    else:
        dataset = dataset["train"]
        # shuffle, we assume the seed is fixed for now
        # we also hardcode other steps for simplicity
        dataset = dataset.shuffle(seed=42)
        train_indices = int(len(dataset) * (1 - 0.2))
        dataset = dataset.select(range(train_indices, len(dataset)))

    manifest = build_and_save_manifest(
        prompt_file=args.prompt_file,
        model=args.model,
        evaluation_type=args.evaluation_type,
        adapter_path=args.adapter_path,
    )

    run_evaluation_for_manifest(manifest, dataset)
    build_and_save_classification_report(manifest=manifest)


if __name__ == "__main__":
    main()

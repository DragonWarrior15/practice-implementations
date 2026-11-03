"""Helper script to run end to end evaluations."""

# some objects to define data exchanges across different functions
import argparse
import logging
from datetime import datetime, timezone

from datasets import load_from_disk
from helpers.constants import (
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


class EvaluationResult(BaseModel):
    id: int
    input: str
    actual: str
    predicted: str


def build_and_save_manifest(prompt_file: str, model: str) -> Manifest:
    """Build and saves the manifest."""
    # fix the time
    now_dt = datetime.now(timezone.utc)

    manifest = Manifest(
        run_id=now_dt.strftime(r"%Y-%m-%dT%H%M%SZ"),
        prompt_file=prompt_file,
        model=model,
        started_at=now_dt.isoformat(),
    )

    run_directory = RUNS_DIRECTORY / manifest.run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    run_directory.joinpath(MANIFEST_FILE_NAME).write_text(
        manifest.model_dump_json(indent=2)
    )

    # return the manifest for downstream consumption
    return manifest


def run_evaluation_for_manifest(manifest: Manifest) -> None:
    """Run inference and persist each case result incrementally."""
    # for each case in the evaluation dataset, generate the outputs and build
    # the EvaluationResult instance
    run_directory = RUNS_DIRECTORY.joinpath(manifest.run_id)
    results_path = run_directory.joinpath(EVALUATION_RESULTS_FILE_NAME)

    model, tokenizer = get_model_and_tokenizer(manifest.model)

    # read the test dataset
    test = load_from_disk(DATA_DIR)["test"]

    # get the labels list
    labels = read_labels()

    with results_path.open("a") as results_file:
        for idx, row in enumerate(test.select(range(2))):
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

    args = parser.parse_args()

    manifest = build_and_save_manifest(
        prompt_file=args.prompt_file, model=args.model
    )

    run_evaluation_for_manifest(manifest)


if __name__ == "__main__":
    main()

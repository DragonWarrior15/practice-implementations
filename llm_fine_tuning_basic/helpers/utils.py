"""Helper utilities for classifier."""

import json

from helpers.constants import LABELS_FILE_PATH, PROMPT_DIR
from mlx.nn import Module
from mlx_lm import generate, load
from mlx_lm.tokenizer_utils import TokenizerWrapper


def read_labels() -> list[str]:
    return json.loads(LABELS_FILE_PATH.read_text())


def get_system_prompt(prompt_file_name: str) -> str:
    """Helper to get read the system prompt from file."""
    prompt_full_path = PROMPT_DIR.joinpath(prompt_file_name)
    labels = read_labels()
    return prompt_full_path.read_text().format(label_names=labels)


def form_messages(system_prompt: str, user_query: str) -> list[dict]:
    """Helper to form the messages for chat input."""
    return [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + [
        {
            "role": "user",
            "content": user_query,
        }
    ]


def get_response_from_llm(
    user_query: str,
    prompt_file_name: str,
    model: Module,
    tokenizer: TokenizerWrapper,
) -> str:
    """Run the loop for getting the triage output.

    We intentionally pass both model and tokenizer, since we are using the
    models directly from mlx_lm.
    """

    system_prompt = get_system_prompt(prompt_file_name)
    messages = form_messages(system_prompt, user_query)

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    text = generate(
        model, tokenizer, prompt=prompt, verbose=False, max_tokens=16
    )

    return text


def get_model_and_tokenizer(model: str) -> tuple[Module, TokenizerWrapper]:
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-bf16")
    return model, tokenizer

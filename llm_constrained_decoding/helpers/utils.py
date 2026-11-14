"""Helper utilities for classifier."""

import json
from typing import Any

from helpers.constants import LABELS_FILE_PATH, MODEL_NAME, PROMPT_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_labels() -> list[str]:
    return json.loads(LABELS_FILE_PATH.read_text())


def get_system_prompt(prompt_file_name: str) -> str:
    """Helper to get read the system prompt from file."""
    prompt_full_path = PROMPT_DIR.joinpath(prompt_file_name)
    labels = "\n".join(read_labels())
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
    model: Any,
    tokenizer: Any,
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
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # for use later
    input_length = prompt["input_ids"].shape[1]

    outputs = model.generate(**prompt, max_length=input_length + 16)

    # slice off the input tokens to only get the assistant reply

    response = tokenizer.decode(
        outputs[0][input_length:], skip_special_tokens=True
    )

    return response


def get_model_and_tokenizer(model_name: str = MODEL_NAME) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )
    return model, tokenizer


def test_run():
    """For a quick end to end run."""
    model, tokenizer = get_model_and_tokenizer()
    response = get_response_from_llm(
        user_query="where is my card",
        prompt_file_name="prompt_v1.md",
        model=model,
        tokenizer=tokenizer,
    )
    print(response)

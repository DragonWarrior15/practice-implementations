"""Helper utilities for classifier."""

import torch
from helpers.constants import MODEL_NAME, TEST_MESSAGE
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def get_tokens_from_messages(
    messages: list[dict[str, str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool = False,
) -> Tensor:
    prompt = tokenizer.apply_chat_template(
        messages,
        # our messages might already contain assistant role, hence set the
        # supplied value
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    return prompt["input_ids"]


def get_log_softmax_from_llm_for_token_sequence(
    tokens: Tensor,
    model: PreTrainedModel,
) -> Tensor:
    """Forward pass on messages to get the logits from LLM."""
    # run the model in eval mode, useful since we are only
    # running inference and not training
    model.eval()

    # disable gradient tracking
    with torch.inference_mode():
        output = model(tokens)
    output_logits: Tensor = output.logits

    # tensor is shape batch size X input sequence length X vocab size
    # we first need to align this with the right tokens
    # the tensor at index 0 is basically the prediction for the token at index
    # 1 and so on.. the last tensor is the prediction after the full sequence
    # is processed, which is not really useful to us unless we are doing
    # a generation exercise
    # shift the tensors
    output_logits = output_logits[:, :-1, :]
    target_ids = tokens[:, 1:]

    output_log_softmax = torch.log_softmax(
        output_logits,
        dim=-1,
        # use extra precision since the model is float16
        dtype=torch.float32,
    )

    # select relevant indices on the vocab part of tensor using unsqueeze
    # unsqueeze add an extra dimension to the array with value 1
    # squeeze removes a redundant dimension, in our case once we have selected
    # exactly a single index for each position of token, the last dimension
    # is effectively empty
    # gather is used to select a single value from the vocab dimension (last)
    relevant_output = torch.gather(
        output_log_softmax, dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)

    return relevant_output


def get_score_from_log_softmax(output: Tensor) -> Tensor:
    """Sum the last dimension to get the sequence score."""
    return output.sum(-1)


def get_model_and_tokenizer(
    model_name: str = MODEL_NAME,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )
    return model, tokenizer


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Helper to get the tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_score_for_assistant_message(
    messages: list[dict[str, str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> Tensor:
    """A helper to run all steps of getting the assistant msg log scores."""
    tokens = get_tokens_from_messages(
        messages=messages,
        model=model,
        tokenizer=tokenizer,
        add_generation_prompt=False,
    )

    # find the token length except last assistant message
    tokens_input_only = get_tokens_from_messages(
        messages=messages[:-1],
        model=model,
        tokenizer=tokenizer,
        # True bec in real use cases we add the assistant token to indicate to
        # the model to start generating the outputs
        add_generation_prompt=True,
    )

    output = get_log_softmax_from_llm_for_token_sequence(
        tokens=tokens,
        model=model,
    )

    # keep only the relevant trailing scores
    output = output[:, -(tokens.shape[1] - tokens_input_only.shape[1]) :]

    score = get_score_from_log_softmax(output)

    return score


def test_run():
    """For a quick end to end run."""
    model, tokenizer = get_model_and_tokenizer()

    tokens = get_tokens_from_messages(
        messages=TEST_MESSAGE, model=model, tokenizer=tokenizer
    )

    output = get_log_softmax_from_llm_for_token_sequence(
        tokens=tokens,
        model=model,
    )

    print(get_score_from_log_softmax(output))

    print(output)


def test_run_2():
    model, tokenizer = get_model_and_tokenizer()
    print(
        get_score_for_assistant_message(
            messages=TEST_MESSAGE, model=model, tokenizer=tokenizer
        )
    )

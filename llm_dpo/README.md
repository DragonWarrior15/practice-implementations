## Direct Preference Optimization (DPO)

### What it does

Direct Preference Optimization (DPO) fine-tunes a language model using pairs of preferred and less-preferred responses to the same prompt.
It encourages the model to favor preferred responses relative to a reference model, without training a separate reward model.

### Steps
1. Setup
    - `cd` to `llm_dpo`
    - Initialize env using `uv sync`
    - Activate env using `source .venv/bin/activate`
    - Set Python Path using `export PYTHONPATH=$(pwd)`
2. Download data
    - `python helpers/prepare_data.py --step download`
    - Inspect statistics using `python helpers/prepare_data --step get_dataset_stats`
    - ```python
        DatasetDict({
            train: Dataset({
                features: ['chosen', 'rejected', 'score_chosen', 'score_rejected'],
                num_rows: 62135
            })
            test: Dataset({
                features: ['chosen', 'rejected', 'score_chosen', 'score_rejected'],
                num_rows: 1000
            })
        })
        Total training data rows:  62135
        Training Data Column Names
        ['chosen', 'rejected', 'score_chosen', 'score_rejected']
        Training Data Example 0:
        Length
        2
        These are messages, printing roles and first 10 chars of content
        0 user Use the pygame libra
        1 assistant Sure, I'd be happy t
        Training Data Example 0: Scores
        score_chosen:  6.0
        score_rejected:  4.0
        Total test data rows:  1000
        ```
3. Understanding a few basics related to the transformers related functions
    - `tokenizer.apply_chat_template` returns a dictionary, whose keys are `input_ids`, `attention_mask`
        ```python
        {
            "input_ids": tensor([[128000, 128006]], device="<device-name>"),
            "attention_mask": tensor([[1, 1]], device="<device-name>"),
        }
        ```
        Reference: [Chat Template](https://huggingface.co/docs/transformers/main/en/chat_templating#using-applychattemplate)
    - `model.forward(**prompt).logits` gives us the logits for the next token at each position for the entire sequence; see the sample below for an input sequence of 57 tokens
        ```python
        tensor([[[  2.8438,   3.5781,   7.0312,  ...,  -1.2422,  -1.2422,  -1.2422],
                [ -3.1719,  -2.2188,  -1.2500,  ...,   2.7656,   2.7656,   2.7656],
                [  3.2500,   5.6875,   3.8750,  ...,  -0.8047,  -0.8047,  -0.8047],
                ...,
                [-10.8750, -13.6875,  -6.6875,  ...,   9.3125,   9.3125,   9.3125],
                [ -0.2383,  -0.2988,  -3.0312,  ...,  -0.0266,  -0.0278,  -0.0272],
                [  3.6875,   6.9375,   7.4062,  ...,  -0.2324,  -0.2324,  -0.2324]]],
            device='mps:0', dtype=torch.bfloat16, grad_fn=<LinearBackward0>)
        ```
        with the shape `torch.Size([1, 57, 128256])`
        
        For each of those 57 positions, the model produces 128,256 scores, one for each possible next token. The logits at position 0 predict the token at input position 1, and so on. The final logits predict a token after the 57-token input.

        Suppose we wanted to score this entire sequence from the model logits perspective, the first 56 tensors will be useful to us, and we would select a single number at each position by first shifting the tensors to align with the original sequence, and then picking the right index at that position.
    - The below two operations allow us to select the relevant token ids after accounting for the shifts
        - `output_logits = output_logits[:, :-1, :]`
        - `target_ids = tokens[:, 1:]`
        - Then we can use `torch.gather` to simply retain the right token id value.
    - `python -c "from helpers.utils import test_run; test_run()"` will produce `-315.1342` as output for our sample message.

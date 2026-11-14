## Basic LLM Constrained Decoding

### What it does

Constrained Decoding is useful when we want to force the LLM output. Suppose we are trying to run a classification task where the LLM output is constrained to a specific set of tokens, in that case we can use constrained decoding so that LLM outputs dont run here and there. Often, LLM might start the response with a different token compared to the the one we want, maybe its trying to form language instead of just giving a response. Constrained Decoding can help in that case.

### Steps
1. Setup
    - `cd` to `llm_constrained_decoding`
    - Initialize env using `uv sync`
    - Activate env using `source .venv/bin/activate`
    - Set Python Path using `export PYTHONPATH=$(pwd)`
2. Download and run the model
    - We will run the model `mlx-community/Llama-3.2-1B-Instruct-bf16` in this example
    - `python helpers/run_model.py`
    - Should ideally output `232`, however the output might change across runs
3. Download the Banking77 Data
    - `python helpers/prepare_data.py --step download`
    - `python helpers/prepare_data.py --step save_labels`

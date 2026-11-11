## Basic LLM Constrained Decoding

### What it does

Constrained Decoding is useful when we want to force the LLM output. Suppose we are trying to run a classification task where the LLM output is constrained to a specific set of tokens, in that case we can use constrained decoding so that LLM outputs dont run here and there. Often, LLM might start the response with a different token compared to the the one we want, maybe its trying to form language instead of just giving a response. Constrained Decoding can help in that case.

### Steps
1. Setup
    - `cd` to `llm_constrained_decoding`
    - Initialize env using `uv sync`
    - Activate env using `source .venv/bin/activate`
    - Set Python Path using `export PYTHONPATH=$(pwd)`

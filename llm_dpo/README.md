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

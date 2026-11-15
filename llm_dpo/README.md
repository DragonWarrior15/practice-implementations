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

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
4. Test the helper functions with an end to end run
    - `python -c "from helpers.utils import test_run; test_run()"`
    - We can expect `card_about_to_expire` as the output
5. The raw model outputs are available under `llm_fine_tuning_basic/runs/2026-11-03T104734Z`
    - We will use those as baselines, and generate the constrained decoding outputs here, and the compare them with the baseline and fine tuned version too
    - Fine tuned version outputs are saved under `llm_fine_tuning_basic/runs/2026-11-04T162523Z`
6. Some updates to `utils.py` to add the Constrained Decoding Logic
    - Along with an Early Stopping Criteria
    - Its basic premise is that if we have figured out that only a single sequence is possible, we can stop generation, do a matching against allowed sequences of tokens and just return that
    - In our example with the question `where is my card`, the constrained decoding run took 6 turns while the constrained decoding + stopping criteria approach took only 3 turns to generate the output.
7. Run evaluations using `python helpers/run_evaluations.py --evaluation_type test`


### Test results: constrained decoding vs. fine-tuning

These runs evaluated the same 3,080 BANKING77 test examples, with the same
`prompt_v1.md` contents and Llama 3.2 1B Instruct BF16 model. The fine-tuned
run uses the 1,000-step LoRA adapter selected on the validation set.

| Metric | Base, prompt only | Base, constrained decoding | Fine-tuned, prompt only |
| --- | ---: | ---: | ---: |
| Accuracy | 18.34% | 26.33% | 77.56% |
| Correct predictions | 565 | 811 | 2,389 |
| Macro precision | 41.01% | 40.84% | 79.71% |
| Macro recall | 18.34% | 26.33% | 77.56% |
| Macro F1 | 20.58% | 24.45% | 76.94% |
| Invalid-label rate | 47.27% | 0.00% | 0.16% |

The constrained run returned a valid label for every example, and its accuracy
was 7.99 percentage points higher than the prompt-only base run. Valid output
does not imply a correct intent: 73.67% of its predictions were still wrong.
The fine-tuned model achieved much higher accuracy on this test set.

Accuracy counts invalid predictions as incorrect; macro metrics weight each
intent equally.

Reports: [constrained base model](runs/2026-11-14T124717Z/classification_report.json)
· [prompt-only base model](../llm_fine_tuning_basic/runs/2026-11-03T104734Z/classification_report.json)
· [fine-tuned model](../llm_fine_tuning_basic/runs/2026-11-04T162523Z/classification_report.json).

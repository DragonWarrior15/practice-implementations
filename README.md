# practice-implementations

Practice implementations of different systems.

## List of Systems

### [Voice Activity Detection System](voice_activity_detection_system/README.md)

- Detects speech-like regions in PCM WAV audio using short-frame RMS energy.
- Applies fixed or adaptive thresholds, then merges short gaps and removes short regions.
- Evaluates frame-level predictions, speech intervals, and runtime against a labeled fixture.

### [LLM Fine Tuning Basic](llm_fine_tuning_basic/README.md)

- Fine-tunes Llama 3.2 1B Instruct on BANKING77 intent classification with MLX-LM and LoRA.
- Prepares chat-format training data and compares adapter checkpoints on a validation set.
- Evaluates the base and fine-tuned models on held-out examples using accuracy, class metrics, and invalid-label rate.

### [LLM Constrained Decoding](llm_constrained_decoding/README.md)

- Uses Hugging Face Transformers to classify BANKING77 requests with the base Llama model.
- Implements a logits processor that restricts generation to tokens that continue an allowed intent label.
- Stops early when only one label remains possible, then compares results with the base and fine-tuned runs.

### [LLM Direct Preference Optimization (DPO)](llm_dpo/README.md)

- Applies DPO with TRL and LoRA to Llama 3.2 1B Instruct using UltraFeedback preference pairs.
- Scores chosen and rejected responses using summed and per-token log-probabilities.
- Compares the base model with checkpoints from a run stopped after 60 steps on 256 validation pairs.
- Observed only small changes in preference accuracy; the results do not establish a meaningful improvement from DPO.

"""Run a small DPO + LoRA smoke test on UltraFeedback.

Run from the project root after downloading the dataset:
    python helpers/train_dpo.py

This checks that model loading, preference-data preparation, and a few training
steps work together. It is not intended to produce a fully trained model.
"""

from datasets import load_from_disk
from helpers.constants import DPO_DIR, MODEL_NAME
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

# Split only the original training set; keep the original test set for the
# final before/after evaluation. The seed makes the split reproducible.
dataset = load_from_disk("data")
train = dataset["train"].shuffle(seed=42)
train_indices = int(len(train) * 0.8)
train, valid = (
    train.select(range(train_indices)),
    train.select(range(train_indices, len(train))),
)

# Keep preprocessing and the first run small. The validation subset is
# separate from training, but is not evaluated automatically in this smoke test.
# train = train.select(range(64))
valid = valid.select(range(256))

training_args = DPOConfig(
    output_dir=DPO_DIR,
    per_device_train_batch_size=1,
    max_steps=100,  # Optimizer steps, not epochs; overrides num_train_epochs.
    logging_steps=1,
    save_strategy="steps",
    save_steps=20,
    max_length=2048,
    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=1,
)

# Freeze the base model and train low-rank adapters on attention query/value
# projections. TRL uses the starting model as the DPO reference with PEFT.
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

trainer = DPOTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=train,
    eval_dataset=valid,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model()

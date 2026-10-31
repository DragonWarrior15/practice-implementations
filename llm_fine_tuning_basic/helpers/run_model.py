"""Demonstrate how to download and run a model for mlx."""

from mlx_lm import generate, load

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-bf16")

prompt = "What is the value of 16 * 32. Just the answer."

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

text = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=16)

print(text)

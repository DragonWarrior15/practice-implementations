"""Demonstrate how to download and run a model."""

from helpers.constants import MODEL_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype="auto"
)

prompt = "What is the value of 8 * 29. Just the answer."

# define messages as a list of dictionaries, with role system and user
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# for use later
input_length = prompt["input_ids"].shape[1]

outputs = model.generate(**prompt, max_length=input_length + 16)

# slice off the input tokens to only get the assistant reply

response = tokenizer.decode(
    outputs[0][input_length:], skip_special_tokens=True
)

print(response)

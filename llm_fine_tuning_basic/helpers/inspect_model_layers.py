from mlx_lm import load
from mlx.utils import tree_flatten

# 1. Load your model
model_name = "mlx-community/Llama-3.2-1B-Instruct-bf16"
model, tokenizer = load(model_name)

print(f"# Model: `{model_name}`\n")


# 2. Print high-level structural components and parameter counts
def print_layer_info(name, parameters):
    params = sum(value.size for _, value in tree_flatten(parameters))
    print(f"| {name} | {params:,} |")


# Check main building blocks
print("## Parameter Counts\n")
print("| Component | Parameters |")
print("| --- | ---: |")
print_layer_info("Embedding Layer", model.model.embed_tokens.parameters())
print_layer_info(
    "Transformer Blocks", [layer.parameters() for layer in model.model.layers]
)
print_layer_info("Final Norm Layer", model.model.norm.parameters())

# Llama 3.2 reuses the embedding weights for its output projection.
print("| LM Head (Tied) | 0 additional |")
print_layer_info("Total Model", model.parameters())
print("\nThe output head shares the embedding weights; they are counted once.")

args = model.args
print("\n## Basic Model Structure\n")
print(f"""```text
Token IDs
  -> Token embedding ({args.vocab_size:,} tokens -> {args.hidden_size:,} features)
  -> Transformer block x {len(model.model.layers)}
       1. RMSNorm -> Self-attention -> Add residual
       2. RMSNorm -> Feed-forward MLP (SwiGLU) -> Add residual
  -> Final RMSNorm
  -> Output projection (shared embedding weights)
  -> {args.vocab_size:,} vocabulary logits per token

Self-attention: q_proj, k_proj, v_proj -> RoPE on Q/K -> causal attention -> o_proj
MLP: down_proj(SiLU(gate_proj(x)) * up_proj(x))
Attention heads: {args.num_attention_heads} query, {args.num_key_value_heads} key/value
MLP intermediate size: {args.intermediate_size:,}
```
""")

# 3. Deep dive into a single Transformer Block (Layer 0) to see internal matrices
print("## Transformer Block 0\n")
layer_0 = model.model.layers[0]
print("| Component | Parameters |")
print("| --- | ---: |")
print_layer_info("Block 0 Total", layer_0.parameters())
print_layer_info("Input RMSNorm", layer_0.input_layernorm.parameters())
print_layer_info("Post-Attn RMSNorm", layer_0.post_attention_layernorm.parameters())


def print_matrix_info(name, layer):
    weight = layer.weight
    print(
        f"| {name} | {weight.size:,}"
        f" | `{weight.shape}` | `{weight.dtype}` |"
    )


print("\nMatrix shapes are `(output features, input features)`.")

# Print Attention matrices
print("\n### Self-Attention Matrices\n")
print("| Matrix | Parameters | Shape | Dtype |")
print("| --- | ---: | --- | --- |")
for name, sub_layer in [
    ("Query (q_proj)", layer_0.self_attn.q_proj),
    ("Key (k_proj)", layer_0.self_attn.k_proj),
    ("Value (v_proj)", layer_0.self_attn.v_proj),
    ("Output (o_proj)", layer_0.self_attn.o_proj),
]:
    print_matrix_info(name, sub_layer)

# Print MLP matrices
print("\n### Feed-Forward (MLP) Matrices\n")
print("| Matrix | Parameters | Shape | Dtype |")
print("| --- | ---: | --- | --- |")
for name, sub_layer in [
    ("Gate (gate_proj)", layer_0.mlp.gate_proj),
    ("Up (up_proj)", layer_0.mlp.up_proj),
    ("Down (down_proj)", layer_0.mlp.down_proj),
]:
    print_matrix_info(name, sub_layer)

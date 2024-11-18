import os
from transformers import AutoTokenizer, Gemma2ForCausalLM
from transformers.cache_utils import HybridCache
import torch

# Disable parallelism in tokenizers for single-threaded execution
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optimize matrix multiplication precision for faster computation
torch.set_float32_matmul_precision("high")

# Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = Gemma2ForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    torch_dtype=torch.bfloat16  # Use bfloat16 for faster computation
)

# Move the model to the CPU
device = torch.device("cpu")
model.to(device)

# Apply `torch.compile` to the model's forward pass
model.forward = torch.compile(
    model.forward,
    mode="reduce-overhead",
    fullgraph=True
)

# Set up the input prompt
input_text = "The theory of special relativity states "
model_inputs = tokenizer(input_text, return_tensors="pt").to(device)
prompt_length = model_inputs.input_ids.shape[1]

# Set up the key/value cache
past_key_values = HybridCache(
    config=model.config,
    max_batch_size=1,
    max_cache_len=model.config.max_position_embeddings,
    device=model.device,
    dtype=model.dtype
)

# Enable passing the key/value cache to the generate method
model._supports_cache_class = True
model.generation_config.cache_implementation = None

# Two warm-up steps to maximize `torch.compile` optimizations
print("Performing warm-up steps...")
for idx in range(2):
    outputs = model.generate(
        **model_inputs,
        past_key_values=past_key_values,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=128
    )
    past_key_values.reset()

# Fast inference step
print("Running fast inference...")
outputs = model.generate(
    **model_inputs,
    past_key_values=past_key_values,
    do_sample=True,
    temperature=1.0,
    max_new_tokens=128
)

# Decode and print the generated output
print("\n=== Model Output ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

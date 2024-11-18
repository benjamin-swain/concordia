import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Specify the model name
    model_name = "google/gemma-2-9b-it"

    # Load the tokenizer and model
    print("Loading tokenizer and model. This may take a while...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for faster computations
        device_map=None            # Ensure the model is not mapped to a GPU
    )

    # Move the model to the CPU
    device = torch.device("cpu")
    model.to(device)

    # Test prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you write a short poem about the stars?"},
    ]

    # Convert messages to input format (Hugging Face does not natively support chat templates)
    input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    # Tokenize input
    print("Tokenizing input...")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Generate output
    print("Generating response...")
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,          # Limit the length of the output
        temperature=0.7,             # Sampling temperature
        top_k=50,                    # Top-k sampling for diversity
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== Model Output ===")
    print(response)

if __name__ == "__main__":
    main()

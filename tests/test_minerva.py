from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Replace with your checkpoint name or path
checkpoint = "/leonardo_scratch/large/userexternal/lcolosi0/minerva-pretraining/huggingface-cpt/minerva-replica-test-scheduler/huggingface/ba500"  # can be local dir or Hugging Face model ID

print(f"Loading model and tokenizer from {checkpoint}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input prompt
prompt = "Tanto tempo fa in una galassia lontana lontana,"
prompt = "Ciao io sono "
# prompt = "3+3="
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate completion
output_ids = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode and print
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n=== Sentence Completion ===")
print(output_text)

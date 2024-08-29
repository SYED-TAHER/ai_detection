from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained GPT-2 model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def calculate_perplexity(text):
    # Tokenize input text for the model
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        # Generate outputs from the model
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

def detect_ai_generated(text):
    perplexity = calculate_perplexity(text)
    threshold = 20  # Example threshold
    is_ai_generated = perplexity < threshold
    return {
        "is_ai_generated": is_ai_generated,
        "perplexity": perplexity
    }

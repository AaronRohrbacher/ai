import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use other GPT-2 variants like "gpt2-medium", "gpt2-large", or "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Your custom training text
# Specify the path to your text file
file_path = 'training.txt'

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    training_text = file.read() [0:1024]

# Tokenize the training text
input_ids = tokenizer.encode(training_text, return_tensors="pt")

# Fine-tune the GPT-2 model on your custom text
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
for epoch in range(num_epochs):
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2")

# Example: Generate text using the fine-tuned model
# def generate_text(prompt, model, tokenizer, max_length=100):
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#     output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     return generated_text

# # Example prompt
# prompt = "Once upon a time"

# # Generate text using the fine-tuned GPT-2 model
# generated_text = generate_text(prompt, model, tokenizer)
# print(generated_text)
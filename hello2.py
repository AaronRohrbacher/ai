import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use other GPT-2 variants like "gpt2-medium", "gpt2-large", or "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Specify the path to your text file
file_path = 'training.txt'

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    text_content = file.read()

# Set hyperparameters
max_seq_length = 512  # Adjust as needed
stride = 100  # Adjust as needed
num_epochs = 3  # Adjust as needed

# Break the text into smaller chunks
chunks = [text_content[i:i + max_seq_length] for i in range(0, len(text_content), stride)]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(num_epochs):
    for chunk in chunks:
        # Tokenize the chunk
        input_ids = tokenizer.encode(chunk, return_tensors="pt")

        # Forward pass
        outputs = model(input_ids, labels=input_ids)

        # Calculate loss and backpropagation
        loss = outputs.loss
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

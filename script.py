import os
import docx
import re
from datasets import Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import gc
import torch

# Function to read text from .docx files
def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Path to the data folder
data_dir = './Oferte test'

# Debug message to check directory
print(f"Checking directory: {data_dir}")
if not os.path.exists(data_dir):
    print(f"Directory does not exist: {data_dir}")
else:
    print(f"Directory exists: {data_dir}")
    files = os.listdir(data_dir)
    if not files:
        print("No files in directory")
    else:
        print(f"Files in directory: {files}")

# Read all documents and combine texts
documents = []

for filename in os.listdir(data_dir):
    if filename.endswith('.docx'):
        file_path = os.path.join(data_dir, filename)
        try:
            documents.append(read_docx(file_path))
            print(f"Read file: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Combine all documents into a single text
corpus = "\n".join(documents)

# Debug message to check the length of the corpus
print(f"Combined corpus length: {len(corpus)}")

# Clean the text
corpus = re.sub(r'\s+', ' ', corpus).strip()

# Save corpus to a text file for training
corpus_path = './corpus.txt'
with open(corpus_path, 'w', encoding='utf-8') as f:
    f.write(corpus)

# Debug message to confirm saving
print(f"Corpus saved to {corpus_path}")

# Load the text file as a dataset
dataset = Dataset.from_text(corpus_path)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)

# Use the Meta-Llama-3-8B model
model_name = "meta-llama/Meta-Llama-3-8B"

# Add your Hugging Face API token here
api_token = "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"

# Login to Hugging Face
from huggingface_hub import login
login(api_token)

# Replace use_auth_token with token
tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Garbage collection
gc.collect()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments with gradient checkpointing enabled
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,  
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    gradient_checkpointing=True, 
)

# Define the model with gradient checkpointing
model = AutoModelForCausalLM.from_pretrained(model_name, token=api_token)
model.gradient_checkpointing_enable()

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./results")
tokenizer.save_pretrained("./results")

# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("./results")

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move the model to GPU, if available
model.to(device)

# Create text generation pipeline and set device
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Prompt based on client's request
prompt = """
Clientul solicită dezvoltarea unei aplicații de tip Glovo. Specifică următoarele secțiuni:
● Sistem de logistică a rider-ului.
● Înregistrarea restaurantelor în aplicație.
● Primirea comenzilor de către clienți.
● Posibilitatea de a adăuga sau elimina produse din meniul restaurantului.
● Plata cu cardul.
Task-uri Adiționale Ne-meneționate de Client, dar Necesare:
● Secțiunea financiară: cum restaurantele solicită bani de la administratorii aplicației.
● Modalități de plată pentru rideri.
● Generarea automată a facturilor și posibilitatea clientului de a descărca factura generată.
"""

# Generate text
generated_text = generator(prompt, max_length=500, num_return_sequences=1, truncation=True)

# Display generated text
print(generated_text[0]['generated_text'])

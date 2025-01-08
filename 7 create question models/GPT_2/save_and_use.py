import os
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

# Load data
input_dir = "input_texts/"  # Directory containing input text files
output_dir = "output_texts/"  # Directory containing output text files

input_files = [f"input_{i}.txt" for i in range(21000)]
output_files = [f"output_{i}.txt" for i in range(21000)]

input_texts = []
output_texts = []

for input_file, output_file in zip(input_files, output_files):
    with open(os.path.join(input_dir, input_file), "r", encoding="utf-8") as f:
        input_texts.append(f.read().strip())
    with open(os.path.join(output_dir, output_file), "r", encoding="utf-8") as f:
        output_texts.append(f.read().strip())

# Combine inputs and outputs for GPT-2 format
combined_texts = [f"{inp}\n\n{out}" for inp, out in zip(input_texts, output_texts)]

# Prepare data for training
train_texts, val_texts = train_test_split(combined_texts, test_size=0.2, random_state=42)

train_data = Dataset.from_dict({"text": train_texts})
val_data = Dataset.from_dict({"text": val_texts})

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

train_dataset = train_data.map(preprocess_function, batched=True)
val_dataset = val_data.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Evaluate on validation data
def compute_metrics(predictions, references):
    rouge = Rouge()
    bleu_scores = []

    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_refs = [tokenizer.decode(ref, skip_special_tokens=True) for ref in references]

    # Compute ROUGE
    rouge_scores = rouge.get_scores(decoded_preds, decoded_refs, avg=True)

    # Compute BLEU
    references_bleu = [[ref.split()] for ref in decoded_refs]
    predictions_bleu = [pred.split() for pred in decoded_preds]
    bleu_score = corpus_bleu(references_bleu, predictions_bleu)

    return rouge_scores, bleu_score

val_preds = trainer.predict(val_dataset)
predictions = val_preds.predictions
references = val_preds.label_ids

rouge_scores, bleu_score = compute_metrics(predictions, references)
print("ROUGE Scores:", rouge_scores)
print("BLEU Score:", bleu_score)

# Save model and tokenizer
model_save_path = "./gpt2_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Load the saved model and tokenizer for future predictions
def load_model_and_predict(input_text):
    saved_tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
    saved_model = GPT2LMHeadModel.from_pretrained(model_save_path)

    inputs = saved_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = saved_model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    return saved_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example prediction
example_input = "Your input text here for prediction."
predicted_output = load_model_and_predict(example_input)
print("Predicted Output:", predicted_output)

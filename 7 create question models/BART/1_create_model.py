import os
import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

N = 21146

# Load data
input_dir = "../../4 top justifications/top_justifications/"  # Directory containing input text files
output_dir = "../../5 actual cross questions/actual_questions/"  # Directory containing output text files

input_files = [f"top_justification_{i}.txt" for i in range(N)]
output_files = [f"true_question_{i}.txt" for i in range(N)]

input_texts = []
output_texts = []

for input_file, output_file in zip(input_files, output_files):
    with open(os.path.join(input_dir, input_file), "r", encoding="utf-8") as f:
        input_texts.append(f.read().strip())
    with open(os.path.join(output_dir, output_file), "r", encoding="utf-8") as f:
        output_texts.append(f.read().strip())

# Prepare data for training
train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
    input_texts, output_texts, test_size=0.2, random_state=42
)

train_data = Dataset.from_dict({"input_text": train_inputs, "output_text": train_outputs})
val_data = Dataset.from_dict({"input_text": val_inputs, "output_text": val_outputs})

# Load BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

def preprocess_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output_text"], max_length=150, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_data.map(preprocess_function, batched=True)
val_dataset = val_data.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True,
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

# Save model and tokenizer
model_save_path = "./BART_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

'''
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
'''
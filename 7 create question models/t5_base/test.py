import os
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


model_save_path = "./t5_base_model"
# Load the saved model and tokenizer for future predictions
def load_model_and_predict(input_text):
    saved_tokenizer = T5Tokenizer.from_pretrained(model_save_path)
    saved_model = T5ForConditionalGeneration.from_pretrained(model_save_path)

    inputs = saved_tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = saved_model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4)
    return saved_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example prediction
example_input = "Your input text here for summarization."
predicted_summary = load_model_and_predict(example_input)
print("Predicted Summary:", predicted_summary)

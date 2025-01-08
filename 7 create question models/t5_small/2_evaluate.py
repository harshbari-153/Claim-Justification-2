#################################################
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
#################################################


N = 21146
model_save_path = "t5_small_model"
output_path = "../../5 actual cross questions/actual_questions/true_question_"
input_path = "../../4 top justifications/top_justifications/top_justification_"

################### BLEU SCORE ##################
def bleu_score(actual_text, generated_text):
    actual_lines = [line.split() for line in actual_text.splitlines()]
    generated_lines = [line.split() for line in generated_text.splitlines()]
    
    total_score = 0
    for actual, generated in zip(actual_lines, generated_lines):
        total_score += sentence_bleu([actual], generated)
    
    avg_bleu_score = total_score / len(generated_lines)
    return round(avg_bleu_score, 3)
#################################################



################# ROUGE SCORE ###################
def rouge_score(actual_text, generated_text):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    actual_lines = actual_text.splitlines()
    generated_lines = generated_text.splitlines()
    
    total_score = 0
    for actual, generated in zip(actual_lines, generated_lines):
        score = scorer.score(actual, generated)
        total_score += score['rougeL'].fmeasure
    
    avg_rouge_score = total_score / len(generated_lines)
    return round(avg_rouge_score, 3)
#################################################



#################################################
def get_file_content(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
  return content
#################################################

'''
# Example usage
actual = """This is a sample sentence.
Another line for testing purposes."""
generated = """This is a sample sentence.
Another testing line for purposes."""


print("BLEU Score:", bleu_score(actual, generated))
print("ROUGE-L Score:", rouge_score(actual, generated))
'''

saved_tokenizer = BartTokenizer.from_pretrained(model_save_path)
saved_model = BartForConditionalGeneration.from_pretrained(model_save_path)
  
  
# Load the saved model and tokenizer for future predictions
def load_model_and_predict(input_text):
  inputs = saved_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
  summary_ids = saved_model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4)
  return saved_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

'''
# Example prediction
example_input = "Your input text here for summarization."
predicted_summary = load_model_and_predict(example_input)
print("Predicted Summary:", predicted_summary)
'''

total_bleu_score = 0
total_rouge_score = 0

for i in range(N):
  input_file_name = input_path + str(i) + ".txt"
  output_file_name = output_path + str(i) + ".txt"
  
  input_str = get_file_content(input_file_name)
  output_str = get_file_content(output_file_name)
  
  total_bleu_score += bleu_score(load_model_and_predict(input_str), output_str)
  total_rouge_score += rouge_score(load_model_and_predict(input_str), output_str)
  
print("Average Bleu Score: " + str(total_bleu_score*100/N))
print("Average Rouge Score: " + str(total_rouge_score*100/N))
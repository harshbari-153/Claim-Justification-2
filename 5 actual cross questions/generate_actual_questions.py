# Enrollment No: p23ds004
# College: NIT Surat, Gujarat, India.
# Course: M. Tech in Data Science (2023-2025)
# Guide: Krupa K. Jariwala
# Final Year Dissertation
# Topic: Claim Justification


N = 21146


################ Import Packages ################
import pandas as pd
import summerize_paragraph as sp
import extract_keywords as ek
#################################################



################# Get Sentence ##################
def get_sentences(index):
  sentences = []
  
  verdict = dataset.iloc[index, 0]
  
  path = "../4 top justifications/top_justifications/top_justification_" + str(index) + ".txt"
  
  with open(path, 'r', encoding='utf-8', errors='ignore') as file:
    
    for line in file:
      sentences.append(line)
      
  return verdict, sentences[0], sentences[1:6]
#################################################






################ Create Vector ##################
def generate_questions(index):
  # extract sentences
  verdict, claim, justifications = get_sentences(index)
  
  justification_summary = sp.summarize_paragraph(justifications)
  claim_keywords = ek.sentence_keywords(claim)
  justification_keywords = ek.sentence_keywords(justification_summary)
  
  if verdict == "true":
    question = "your Justification aligns with claim"
    
  elif verdict == "mostly-true":
    question = "everything is alright, but \"" + claim_keywords + "\" and \"" + justification_keywords + "\" do not match with each other"
    
  elif verdict == "half-true":
    question = "if \"" + claim_keywords + "\" is true then what about \"" + justification_keywords + "\" ?"
    
  elif verdict == "mostly-false":
    question = "but how can you justify \"" + claim_keywords + "\" and \"" + justification_keywords + "\""
    
  elif verdict == "false":
    question = "\"" + claim_keywords + "\" and \"" + justification_keywords + "\" do not align together"
    
  elif verdict == "pants-fire":
    question = "you are openly lying"
    
  else:
    question = "An error occured in program"
    print(question)
  
  
  # open file to write
  file = open("actual_questions\\true_question_" + str(index) + ".txt", "w", encoding='utf-8')
  file.writelines(question)
  file.writelines("\nYour claim is " + verdict)
  file.close()
#################################################



# open dataset
dataset = pd.read_json("..\dataset\politifact_factcheck_data.json", lines = True)




print("Process Begin")
i = 0
while i < N:
  generate_questions(i)
  
  
  i += 1
  # show progress
  if i % 10 == 0:
    print(str(i*100/N) + "% done")


# sucess
print("Vectorization sucessfully done")
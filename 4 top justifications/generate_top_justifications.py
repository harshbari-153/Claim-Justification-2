# Enrollment No: p23ds004
# College: NIT Surat, Gujarat, India.
# Course: M. Tech in Data Science (2023-2025)
# Guide: Krupa K. Jariwala
# Final Year Dissertation
# Topic: Claim Justification


N = 21146
start = 0


################ Import Packages ################
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gensim
from gensim.models import KeyedVectors
import nltk
#################################################



################# Import Models #################
sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
#nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
#################################################



################# Get Sentence ##################
def get_sentences(index):
  sentences = []
  
  temp = dataset.iloc[index, 2]
  sentences.append(temp)
  
  path = "../1 extract justification/justification/j_" + str(index) + ".txt"
  
  with open(path, 'r', encoding='utf-8', errors='ignore') as file:
    
    for line in file:
      sentences.append(line)
      
  return sentences
#################################################



############## Get Close Sentences ##############
def get_top_justifications(claim, justifications, top_n, file_no):
    # Compute embeddings in 384d
    claim_embedding = sim_model.encode(claim, convert_to_tensor=True)
    justification_embeddings = sim_model.encode(justifications, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(claim_embedding, justification_embeddings)
    
    # Get indices of top N similarities
    top_n_indices = np.argsort(similarities.numpy(), axis=1)[0, -top_n:][::-1]
    
    # Sort indices by similarity in descending order
    top_n_indices = sorted(top_n_indices, key=lambda idx: similarities[0, idx], reverse=True)
    
    
    # Store in file
    file_path = "top_justifications\\top_justification_" + str(file_no) + ".txt"
    file = open(file_path, "w", encoding='utf-8')
    file.write(claim+'\n')
    
    for indice in top_n_indices:
      #file.write("\n")
      file.write(justifications[indice])
    
    file.close()
    
    # return distance_vector(claim, justifications, top_n_indices)
#################################################



########## Store Top Justifications #############
def fetch_justifications(index):
  # extract sentences
  sentences = get_sentences(index)
  
  get_top_justifications(sentences[0], sentences[1:], 5, index)
#################################################



# open dataset
dataset = pd.read_json("..\dataset\politifact_factcheck_data.json", lines = True)



i = 18174
while i < N:
  fetch_justifications(i+start)
  
  i += 1
  # show progress
  if i % 50 == 0:
    print(str(i*100/N) + "% done")


# sucess
print("Top Justifications Fetched")
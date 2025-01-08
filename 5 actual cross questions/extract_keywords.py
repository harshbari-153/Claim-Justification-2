import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

# Download necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def sentence_keywords(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # POS tagging
    pos_tags = pos_tag(filtered_words)
    
    # Focus on nouns (NN, NNS), verbs (VB, VBD, etc.), and adjectives (JJ)
    keywords = [word for word, tag in pos_tags if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ')]
    
    # Select 2-3 most relevant words (manual adjustment can be applied)
    if len(keywords) > 3:
      return " ".join(keywords[:4])
    return " ".join(keywords[:])

'''
# Test cases
sentences = [
    "India developed new National Engineering College.",
    "Restaurants in new city gives best food.",
    "I am waiting so long for lunch break."
]

for sentence in sentences:
    summary = sentence_keywords(sentence)
    print(f"Sentence: {sentence}")
    print(f"Summary: {summary}")
    print()
'''
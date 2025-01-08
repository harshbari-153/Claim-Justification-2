import logging
import warnings
from transformers import pipeline



def summarize_paragraph(sentences):

    # Set logging level to ERROR to suppress warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Initialize the summarization pipeline with PEGASUS
    summarizer = pipeline("summarization", model="google/pegasus-xsum")
    
    # Combine sentences into a single paragraph
    text = " ".join(sentences)
    
    # Generate a summary with stricter length constraints
    summary = summarizer(
        text,
        max_length=30,  # Enforce single concise sentence
        min_length=15,
        do_sample=False
    )
    
    # Return the summarized text
    return summary[0]["summary_text"]


'''
# Example usage
example_1 = [
    "India developed a new National Engineering College.",
    "Thousands of Medical seats have been increased in India.",
    "India spent two billion rupees for Secondary Section Scholarship.",
    "Many Indian Education Institutes give admission by taking donations.",
    "Indian civil exam was caught in new paper leak news."
]

example_2 = [
    "Restaurants in the new city give the best food.",
    "Hostel mess does not taste so good.",
    "Poor people are very hungry.",
    "I am waiting so long for a lunch break.",
    "Mother bird is feeding her child bird."
]

# Generate summaries
output_1 = summarize_paragraph(example_1)
output_2 = summarize_paragraph(example_2)

print("Example 1 Output:", output_1)
print("Example 2 Output:", output_2)
'''
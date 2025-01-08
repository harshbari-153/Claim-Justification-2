# Cross Question Generator in Claim Justification

## Overview
Many times in the real world, we encounter claims with inadequate or irrelevant justifications. Such mismatches can lead to confusion and misinformation. This project aims to address this problem by generating targeted questions that highlight gaps in the justification provided for a given claim.

The Cross Question Generator takes a claim and its corresponding justification as input and outputs a question (a single sentence) that should be asked to address the missing or unclear aspects in the justification.

## Dataset
The model has been trained using the Polifact Dataset, which contains:
- **21,000 claims** with their corresponding justifications.
- Claims and justifications scraped from the Polifact website.

## Models Used
This project fine-tuned five state-of-the-art text generation models to achieve the task of generating cross questions:

1. **BART**
2. **GPT-2**
3. **PEGASUS**
4. **T5-base**
5. **T5-small**

Each fine-tuned model is stored in its respective folder.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/cross-question-generator.git
   cd cross-question-generator
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the fine-tuned models from the respective folders.

## Usage
Run the main script to generate cross questions:
```bash
python generate_question.py --model [model_name] --claim "<claim>" --justification "<justification>"
```
### Example:
```bash
python generate_question.py --model t5_base --claim "Vaccines cause autism." --justification "There are anecdotal reports from parents."
```
**Output:** "What scientific evidence supports the link between vaccines and autism?"

## Features
- Generates cross questions for claims with inadequate justifications.
- Supports five different fine-tuned models for comparison and experimentation.
- Easy to integrate into workflows for claim-justification analysis.

## Model Details
- **BART**: A transformer model pre-trained for text-to-text generation tasks, fine-tuned for generating cross questions.
- **GPT-2**: A generative model capable of understanding and generating coherent text.
- **PEGASUS**: Specialized in abstractive text summarization, adapted here for generating concise questions.
- **T5-base** and **T5-small**: Versatile models fine-tuned for various NLP tasks, leveraged for question generation.

## Dataset Details
The Polifact Dataset used in this project includes claims and their corresponding justifications scraped from the Polifact website. The dataset enables the model to learn patterns in claims and justifications and identify gaps that can be addressed with targeted questions.

## Folder Structure
```
|-- 4 top justifications/
|   |-- (21000 text files containing claims and their justifications)
|-- 5 actual cross questions/
|   |-- (21000 text files containing the desired output questions)
|-- 6 merge justification question/
|   |-- merge_program.py (Script to merge justifications and questions for training)
|-- 7 create question models/
|   |-- bart/
|   |   |-- fine_tuned_model/
|   |   |-- finetune_script.py
|   |   |-- model.zip
|   |-- gpt_2/
|   |   |-- fine_tuned_model/
|   |   |-- finetune_script.py
|   |   |-- model.zip
|   |-- pegasus/
|   |   |-- fine_tuned_model/
|   |   |-- finetune_script.py
|   |   |-- model.zip
|   |-- t5_base/
|   |   |-- fine_tuned_model/
|   |   |-- finetune_script.py
|   |   |-- model.zip
|   |-- t5_small/
|   |   |-- fine_tuned_model/
|   |   |-- finetune_script.py
|   |   |-- model.zip
|-- data/
|   |-- polifact_dataset.csv
|-- generate_question.py
|-- requirements.txt
|-- README.md
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git commit -m "Add your message here"
   git push origin feature/your-feature-name
   ```
4. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Polifact Dataset for providing real-world claims and justifications.
- Hugging Face Transformers library for model fine-tuning and implementation.

## Contact
For questions or suggestions, feel free to contact:
- [Your Name](mailto:your.email@example.com)
- GitHub: [Your GitHub Profile](https://github.com/username)


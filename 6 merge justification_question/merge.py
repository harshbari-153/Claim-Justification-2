import os
import csv

N = 21146

# Define the directories
input_dir = "../4 top justifications/top_justifications"  # Replace with the path to your input files
output_dir = "../5 actual cross questions/actual_questions"  # Replace with the path to your output files
csv_file_path = "./dataset.csv"  # Path for the resulting CSV file

# Initialize a list to store rows for the CSV
data_rows = []

# Loop through the input files
for i in range(N):  # Adjust the range if needed
    input_file_path = os.path.join(input_dir, f"top_justification_{i}.txt")
    output_file_path = os.path.join(output_dir, f"true_question_{i}.txt")
    
    # Check if both files exist and print debug information
    if not os.path.exists(input_file_path):
        print(f"Skipping input file: {input_file_path} does not exist.")
        continue
    if not os.path.exists(output_file_path):
        print(f"Skipping output file: {output_file_path} does not exist.")
        continue

    # Read input file content (keeping newlines)
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        input_content = input_file.read().strip()  # Keep \n characters intact

    # Read output file content (keeping newlines)
    with open(output_file_path, "r", encoding="utf-8") as output_file:
        output_content = output_file.read().strip()  # Keep \n characters intact

    # Append to the list as a new row
    data_rows.append({"input": input_content, "output": output_content})

# Check if data_rows is populated before writing to CSV
print(f"Total rows to write: {len(data_rows)}")

# Write the rows to a CSV file
if data_rows:
    with open(csv_file_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["input", "output"], quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        writer.writeheader()  # Write the header row
        writer.writerows(data_rows)  # Write all data rows
    print(f"CSV file created at: {csv_file_path}")
else:
    print("No data to write to CSV.")
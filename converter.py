# import pandas as pd

# # Read the raw dialogs.csv as a list of lines
# with open('dialogs.csv', 'r', encoding='utf-8') as file:
#     lines = file.readlines()

# # Remove any whitespace/newlines and filter out empty lines
# lines = [line.strip() for line in lines if line.strip()]

# # Extract questions (even indices: 0, 2, 4, ...) and answers (odd indices: 1, 3, 5, ...)
# questions = [lines[i] for i in range(0, len(lines), 2)]  # Even indices
# answers = [lines[i] for i in range(1, len(lines), 2)]    # Odd indices

# # Ensure both lists are the same length (trim if necessary)
# min_length = min(len(questions), len(answers))
# questions = questions[:min_length]
# answers = answers[:min_length]

# # Clean the text (remove extra commas or split on first comma if needed)
# def clean_text(text):
#     if isinstance(text, str):
#         # Split on the first comma and take the first part, or keep the whole string
#         parts = text.split(',', 1)[0].strip()
#         return parts if parts else text
#     return text

# questions = [clean_text(q) for q in questions]
# answers = [clean_text(a) for a in answers]

# # Create a new DataFrame
# new_dataset = pd.DataFrame({
#     'Question': questions,
#     'Answer': answers
# })

# # Save to chatbot_dataset.csv
# new_dataset.to_csv('chatbot_dataset.csv', index=False)
# print("Converted and cleaned dataset saved as 'chatbot_dataset.csv'")
# print(new_dataset.head())  # Preview the first 5 rows

import pandas as pd

# Load the WikiQA dataset (e.g., train set)
wikiqa_data = pd.read_csv('D:\chatbot\wiki_dataset\WikiQA-train.tsv', sep='\t')

# Filter for correct answers (Label == 1)
qa_pairs = wikiqa_data[wikiqa_data['Label'] == 1][['Question', 'Sentence']]

# Rename columns to match your chatbot format
qa_pairs = qa_pairs.rename(columns={'Question': 'Question', 'Sentence': 'Answer'})

# Save to chatbot_dataset.csv
qa_pairs.to_csv('chatbot_dataset.csv', index=False)
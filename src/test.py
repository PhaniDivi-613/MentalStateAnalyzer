


from sklearn.metrics import f1_score
import numpy as np

# Actual labels for a set of documents (multilabel classification)
# Each row represents a document, and each column corresponds to a label
y_true = np.array([
    [1, 0, 1],  # Sports and Health
    [0, 1, 0],  # Technology
    [1, 0, 0],  # Sports
    [0, 1, 1],  # Technology and Health
    [1, 0, 1],  # Sports and Health
])

# Predicted labels for the same set of documents
y_pred = np.array([
    [1, 0, 1],  # Correct prediction
    [0, 1, 0],  # Correct prediction
    [1, 0, 1],  # Correct prediction
    [1, 0, 1],  # Incorrect prediction for Technology
    [1, 0, 0],  # Incorrect prediction for Health
])

# Calculate F1 score without ignoring any label (default behavior)
f1_default = f1_score(y_true, y_pred, average='micro')
print("F1 Score (Default):", f1_default)

# Calculate F1 score by ignoring the "Technology" label
f1_ignore_technology = f1_score(y_true, y_pred, average='micro', labels=[0, 2])
print("F1 Score (Ignore Technology):", f1_ignore_technology)


import json
import torch
from transformers import AutoTokenizer, BertTokenizer, PreTrainedTokenizerBase

model_name = "bert-base-uncased"  # Replace with your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if isinstance(tokenizer, PreTrainedTokenizerBase):
    print("tokenizer is an instance of AutoTokenizer")
else:
    print("tokenizer is not an instance of AutoTokenizer")

# Get the class of the tokenizer
current_class = tokenizer.__class__

# Initialize a list to store the parent classes
parent_classes = []

# Iterate through the class hierarchy and collect parent classes
while current_class is not None:
    parent_classes.append(current_class)
    current_class = current_class.__bases__[0] if current_class.__bases__ else None

# Print the collected parent classes
for cls in parent_classes:
    print(cls)
# data_pairs = [["This is an example sentence.", "phani is bad boy"], ["This is an example sentence.", "phani is bad boy"]]
# token_ids = []
# for sentences in data_pairs:
#     tokenized_sentences = [tokenizer(sentence, truncation=True, max_length=100, return_tensors='pt')['input_ids'] for sentence in sentences]
#     token_ids.append(torch.cat(tokenized_sentences, dim=1).squeeze())

# text_lengths = torch.Tensor([len(text) for text in token_ids])
# max_text_length = max(text_lengths.max().item(), 20)
# print(tokenizer.pad_token_id)
# input_ids = torch.ones(len(token_ids), max_text_length) * tokenizer.pad_token_id
# for i, tokens in enumerate(token_ids):
#         print(tokens)
#         input_ids[i][:len(tokens)] = tokens

# print(input_ids)


# Open the file and read its contents
# with open(filepath, 'r') as file:
#     data = file.read()

# # Split the content into individual JSON objects, one per line
# json_objects = data.strip().split('\n')

# # Parse each JSON object into a list of dictionaries
# list_of_dictionaries = [json.loads(obj) for obj in json_objects]
# print(len(list_of_dictionaries))

# # Serialize the list of dictionaries into a single JSON object
# json_object = json.dumps(list_of_dictionaries)

# 'json_object' is now a single JSON object containing the entire list
# for sample in list_of_dictionaries:
#         # count+=1
#         # if(count==10):
#         #   break
#         # self.labels.append(sample["label"])
#         # self.data.append(sample["posts"])
#         lbl = sample["label"]
#         posts = sample["posts"]
#         print()
#         print("number of posts: ", len(posts))
#         print("post-0", posts[0]["text"]) 
#         print("len of 0 post", len(posts[0]["text"]))
#         print("label", lbl)


# from your_module.vocabulary import Vocabulary

# # Sample text data
# text_data = [
#     "This is a sample sentence.",
#     "Another example for testing.",
#     "We need to build a vocabulary.",
# ]

# # Tokenize the text data
# tokenized_text = [sentence.split() for sentence in text_data]

# # Create a vocabulary
# vocab = Vocabulary()
# for tokens in tokenized_text:
#     for token in tokens:
#         vocab.add_word(token)

# # Finalize the vocabulary (sort words by frequency, limit vocabulary size)
# vocab.finalize(frequency_threshold=1, word_count_limit=10)

# # Convert text to tensors of indices
# text_to_convert = "This is another example."
# tokens = text_to_convert.split()
# tensor = vocab.tokens_to_tensor(tokens)

# print("Vocabulary:")
# print(vocab.words)

# print("\nText to Tensor:")
# print(tokens)
# print(tensor)

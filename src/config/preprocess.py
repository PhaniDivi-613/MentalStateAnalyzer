import os

from .base import ConfigBase

current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
data_directory = os.path.join(parent_directory, "data")

class PreprocessConfig(ConfigBase):
    
    # Default file names for training, validation, and testing datasets
    json_file: str = "twitter-1h1h.json"
    
    # Default directory for dataset storage
    datadir: str = data_directory
    
    # Tokenizer choice for text processing (e.g., for splitting text into words or characters)
    # Possible choices: "space", "char"
    tokenizer: str = "space"

    # Number of words to retain in the vocabulary (-1 indicates no limit on dictionary size)
    nwords: int = -1

    # Minimum word count required for a word to be included in the dictionary
    min_word_count: int = 1

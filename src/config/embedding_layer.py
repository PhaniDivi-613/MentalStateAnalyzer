from typing import Optional

from .base import ConfigBase
from dataset.dictionary import Dictionary

# Define a base configuration class for embedding layers
class EmbeddingLayerConfig(ConfigBase):
    pass

# Define a configuration class for static word embeddings
class StaticEmbeddingLayerConfig(EmbeddingLayerConfig):
    """Static Word Embeddings: word2vec, glove, or randomly initialized vectors"""
    # Set the dimension of word vectors to 300 by default
    dim: int = 300
    
    # Specify the method for embedding initialization, with "random" as the default choice
    method: str = "random"  # Choices: random/pretrained

    # Specify the path to pretrained word vectors (valid when the method is "pretrained")
    # The pretrained vector files are in text format, where each line contains a word and its vector.
    # Values are separated by spaces.
    pretrained_path: Optional[str] = None

    # Dictionary (optional), determined at runtime
    dictionary: Optional[Dictionary] = None

# Define a configuration class for BERT word embeddings
class BertEmbeddingLayerConfig(EmbeddingLayerConfig):
    # Set the dimension of hidden states to 768 by default, which can be 1024 for the "large" model
    dim: int = 768

    # Specify the directory containing the BERT pre-trained model files
    model_dir: str = "bert-base-uncased"

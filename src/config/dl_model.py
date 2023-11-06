from .base import ConfigBase
from .embedding_layer import EmbeddingLayerConfig, StaticEmbeddingLayerConfig
from .classifier import ClassifierConfig, RNNClassifierConfig

class DLModelConfig(ConfigBase):
    """A deep learning model can be divided into two main components:
        1. The embedding layer, which can be Word2Vec, Glove, BERT, or randomly initialized vectors.
        2. The classifier layer, such as TextCNN, fully connected layers, TextRNN, and so on.
    """
    # The default embeddings are staticEmbeddings.
    embedding_layer: EmbeddingLayerConfig = StaticEmbeddingLayerConfig()

    # The default model for classification is RNNClassifier.
    classifier: ClassifierConfig = RNNClassifierConfig()

from typing import Union, Optional, List

from .base import ConfigBase
from .components import RNNConfig

# Define a base configuration class for classifiers
# don't need to specify input_size and output_size; the model will determine them based on the number of labels.
class ClassifierConfig(ConfigBase):
    # input_size represents the dimension of the embedding layer
    input_size: Optional[int] = None

    # output_size is the size of the output space and should correspond to the number of labels
    output_size: Optional[int] = None

# # Define a configuration class for a CNN-based text classifier
# class CNNClassifierConfig(ClassifierConfig):
#     kernel_sizes: List[int] = [2, 3, 4]
#     num_kernels: int = 256
#     top_k_max_pooling: int = 1  # Max top-k pooling.
#     hidden_layer_dropout: float = 0.5

# # Define a configuration class for a linear text classifier
# class LinearClassifierConfig(ClassifierConfig):
#     # Choose "first" or "mean" as the pool_method.
#     # If "first," use the first token's embedding as the input to the Linear layer.
#     # If "mean," average the sequence embeddings as the input to the Linear layer.
#     pool_method: str = "first"
    
#     # Dropout probability for the embedding layer
#     embedding_dropout: float = 0.1
    
#     # Hidden layer settings
#     hidden_units: Optional[List[int]] = None
#     activations: Optional[List[str]] = None
#     hidden_dropouts: Optional[List[float]] = None

# Define a configuration class for an RNN-based text classifier
class RNNClassifierConfig(ClassifierConfig):
    # Configuration for the RNN layer
    rnn_config: RNNConfig = RNNConfig()
    # If True, use an attention mechanism to calculate the output state
    use_attention: bool = False
    # Dropout probability for the context
    dropout: float = 0.2

# # Define a configuration class for a Recurrent Convolutional Neural Network (RCNN) text classifier
# class RCNNClassifierConfig(ClassifierConfig):
#     # Configuration for the RNN layer
#     rnn_config: RNNConfig = RNNConfig()
#     # Size of the latent semantic vector
#     semantic_units: int = 512

# # Define a configuration class for a Deep RNN (DRNN) text classifier
# class DRNNClassifierConfig(ClassifierConfig):
#     # Configuration for the RNN layer
#     rnn_config: RNNConfig = RNNConfig()
#     # Dropout probability applied in the DRNN input and output layers
#     dropout: float = 0.2
#     # Window size for the RNN
#     window_size: int = 10

# # Define a configuration class for a Deep Pyramid Convolutional Neural Network (DPCNN) text classifier
# class DPCNNClassifierConfig(ClassifierConfig):
#     # Kernel size
#     kernel_size: int = 3
#     # Stride of pooling
#     pooling_stride: int = 2
#     # Number of kernels
#     num_kernels: int = 16
#     # Number of blocks for DPCNN
#     blocks: int = 2
#     # Dropout probability on convolutional features
#     dropout: float = 0.2

# # Define a configuration class for a Transformer-based text classifier
# class TransformerClassifierConfig(ClassifierConfig):
#     d_model: int = 512
#     nhead: int = 8
#     num_encoder_layers: int = 6
#     dim_feedforward: int = 2048
#     dropout: float = 0.1
#     layer_norm_eps: float = 1e-5
#     batch_first: bool = True
#     norm_first: bool = False

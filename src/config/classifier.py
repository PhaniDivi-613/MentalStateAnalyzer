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

class RNNClassifierConfig(ClassifierConfig):
    # Configuration for the RNN layer
    rnn_config: RNNConfig = RNNConfig()
    # If True, use an attention mechanism to calculate the output state
    use_attention: bool = False
    # Dropout probability for the context
    dropout: float = 0.2

# Define a configuration class for a Memory Augmented RNN (MRNN) text classifier
class MRNNClassifierConfig(ClassifierConfig):
    #confgiguration for memory Layer
    memory_config: RNNConfig = RNNConfig()

    # Configuration for the RNN layer
    rnn_config: RNNConfig = RNNConfig()

    # Dropout probability applied in the MRNN input and output layers
    dropout: float = 0.2

    # Window size for the RNN
    window_size: int = 10

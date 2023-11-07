from typing import Any, Iterable, List, Optional, Tuple
import os

from .dl_model import DLModelConfig
from .optimizer import OptimizerConfig, AdamConfig
from .scheduler import SchedulerConfig
from .data_loader import DataLoaderConfig
from .base import ConfigBase
from .criterion import CriterionConfig, BinaryCrossEntropyLossConfig
from .scheduler import NoneSchedulerConfig

current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
data_directory = os.path.join(parent_directory, "data")

class DLTrainerConfig(ConfigBase):
    """Training configuration for deep learning model"""

    # Whether to use GPU
    use_cuda: bool = False

    # Training epochs
    epochs: int = 10

    # Specify how to save the best model
    # If score_method is "accuracy," save the model with the highest accuracy on the validation set
    # If score_method is "loss," save the model with the lowest loss
    score_method: str = "accuracy"

    # Specify the directory for saving checkpoints
    ckpts_dir: str = os.path.join(data_directory, "ckpts")

    # Whether to save checkpoints at every epoch
    save_ckpt_every_epoch: bool = True

    # Random seed to ensure reproducibility
    random_state: Optional[int] = 2020

    # Start training from a checkpoint specified by state_dict_file
    # state_dict_file: Optional[str] = "./ckpts/1.pt"
    state_dict_file: Optional[str] = None

    # Whether to load the dictionary and label2id from state_dict_file
    load_dictionary_from_ckpt: Optional[bool] = False

    # Set to True for testing purposes with a small dataset
    test_program: Optional[bool] = False

    # Stop training after a certain number of epochs without improvement in the evaluation metric
    early_stop_after: Optional[int] = None

    # Clip the gradient norm if set
    max_clip_norm: Optional[float] = None

    # Whether to perform evaluation and model selection
    do_eval: bool = True

    # If do_eval is True, specify whether to load the best model state dict after training
    load_best_model_after_train: bool = True

    # Number of batches to print training information
    num_batch_to_print: int = 10

    # Configuration for the optimizer used in parameter updates
    optimizer: OptimizerConfig = AdamConfig()
    scheduler: Optional[SchedulerConfig] = NoneSchedulerConfig()

    # Configuration for the deep learning model
    model: DLModelConfig = DLModelConfig()

    # Configuration for the data loader
    data_loader: DataLoaderConfig = DataLoaderConfig()

    # Configuration for the loss criterion
    criterion: CriterionConfig = BinaryCrossEntropyLossConfig()

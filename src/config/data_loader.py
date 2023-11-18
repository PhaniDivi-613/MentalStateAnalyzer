import os

from .base import ConfigBase

current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
data_directory = os.path.join(parent_directory, "data")

class DataLoaderConfig(ConfigBase):
    json_file: str = os.path.join(data_directory, "twitter-1h1h.json")
    train_file: str = os.path.join(data_directory, "twitter-1h1h.json")
    test_file: str = os.path.join(data_directory, "twitter-1h1h.json")
    valid_file: str = os.path.join(data_directory, "twitter-1h1h.json")
    
    raw_data_path: str = os.path.join(data_directory, "msa.joblib")

    # How many samples per batch to load (default: 32).
    batch_size: int = 32 

    max_len: int = 10

     # Set to True to reshuffle the data at every epoch (default: False).
    shuffle: bool = False 

    # How many subprocesses to use for data loading (0 means loading in the main process, default: 0).
    num_workers: int = 0  

    # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    pin_memory: bool = True 

    # Set to True to drop the last incomplete batch.
    drop_last: bool = True  

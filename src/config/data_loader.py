# from .base import ConfigBase

# class DataLoaderConfig(ConfigBase):
#     json_file_path: str = "../../data/twitter-1h1h.json"

#     # How many samples per batch to load (default: 32).
#     batch_size: int = 32 

#     max_len: int = 512

#      # Set to True to reshuffle the data at every epoch (default: False).
#     shuffle: bool = False 

#     # How many subprocesses to use for data loading (0 means loading in the main process, default: 0).
#     num_workers: int = 0  

#     # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
#     pin_memory: bool = True 

#     # Set to True to drop the last incomplete batch.
#     drop_last: bool = True  

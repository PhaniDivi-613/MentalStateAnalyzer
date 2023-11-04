# #!/usr/bin/env python3
# import random
# import json

# import click
# import torch
# import numpy as np

# from dataset.raw import TextClfRawData
# from textclf.utils.raw_data import save_raw_data
# from textclf.utils.config import config_from_json
# from textclf.utils.create import create_instance
# from textclf.utils.config import help_config_main

# # Define the main function that serves as the entry point for the script.
# @click.group()
# @click.option("--config-file", default="")
# @click.pass_context
# def main(context, config_file):
#     """Configuration settings can be provided from a file or directly in JSON format."""
#     context.obj = {}
#     if config_file:
#         with open(config_file) as f:
#             json_data = json.load(f)
#             context.obj['config'] = config_from_json(json_data)
#     else:
#         context.obj['config'] = None

# # Define a command for printing or generating a configuration with default values.
# @main.command(help="Print/Generate a config with default values.")
# @click.pass_context
# def help_config(context):
#     """Print the configuration for a specified 'class_name' with default values."""
#     help_config_main()

# # Define a command for preprocessing data.
# @main.command(help="Preprocess data.")
# @click.option("--save-path", type=str, default="./textclf.joblib")
# @click.pass_context
# def preprocess(context, save_path):
#     preprocess_config = context.obj['config']
#     raw_data = TextClfRawData(preprocess_config)
#     raw_data.describe()
#     save_raw_data(raw_data, save_path)

# # Define a command for training a model and saving it.
# @main.command(help="Train model and save.")
# @click.pass_context
# def train(context):
#     trainer_config = context.obj['config']

#     # Ensure reproducibility by setting a random seed if provided.
#     if getattr(trainer_config, "random_state", None):
#         print(f"Random State: {trainer_config.random_state}")
#         np.random.seed(trainer_config.random_state)
#         torch.manual_seed(trainer_config.random_state)
#         torch.cuda.manual_seed_all(trainer_config.random_state)
#         torch.backends.cudnn.deterministic = True
#         random.seed(trainer_config.random_state)
#         torch.backends.cudnn.enabled = False
#     trainer = create_instance(trainer_config)
#     trainer.train()

# # Define a command for loading a model and testing it.
# @main.command(help="Load model and test.")
# @click.pass_context
# def test(context):
#     tester_config = context.obj['config']
#     tester = create_instance(tester_config)
#     tester.test()

# # Run the main function if this script is executed directly.
# if __name__ == "__main__":
#     main()

# Import necessary libraries
from typing import List
import sys
import json
import config as config
# from config import (
#     PreprocessConfig,
#     DLTrainerConfig,
#     DLTesterConfig,
# )
from config.base import ConfigBase

# Define functions for working with configuration data in JSON format

# Function to convert JSON data into a configuration dictionary
def json_to_configdict(json_data):
    res = {}
    for k in json_data:
        obj = json_data[k]
        if type(obj) == dict and "__class__" in obj:
            class_name = obj["__class__"]
            res[k] = getattr(config, class_name)(json_to_configdict(obj["params"]))
        else:
            res[k] = obj
    return res

# Function to create a configuration object from JSON data
def config_from_json(json_data):
    class_name = json_data['__class__']
    config_dict = json_to_configdict(json_data["params"])
    return getattr(config, class_name)(config_dict)

# # Function to get the name of a configuration class without the suffix
# def get_instance_name(config, drop_suffix=True):
#     name = config.__class__.__name__
#     return name[:-6] if drop_suffix else name

# # Function to convert a configuration class to a string and remove the suffix
# def stringfy(config_class, drop_suffix=True):
#     name = str(config_class).split('.')[-1]
#     return name[:-8] if drop_suffix else name[:-2]

# # Function to help configure settings interactively
# def help_config_dfs(config, options):
#     config_keys, common_keys = split_keys(config)
#     for key in common_keys:
#         options[key] = getattr(config, key)
#     if not config_keys:
#         return
#     for key in config_keys:
#         print(f"Setting {key}")
#         value = getattr(config, key)
#         base = value.__class__.__base__
#         try:
#             from textclf.utils.common import CONFIG_CHOICES
#             choices = CONFIG_CHOICES[base]
#             choice_id = query_choices_to_user(
#                 [stringfy(c) for c in choices],
#                 get_instance_name(value),
#                 key
#             )
#             if choice_id == -1:
#                 choice = value
#             else:
#                 choice = choices[choice_id]()
#         except KeyError:
#             choice = value

#         options[key] = {}
#         options[key]["__class__"] = choice.__class__.__name__
#         options[key]["params"] = {}
#         help_config_dfs(choice, options[key]["params"])

# # Function to prompt the user to choose from a list of choices
# def query_choices_to_user(choices: List[str], default: str, base: str):
#     print(f"{base} has the following choices (Default: {default}):")
#     for i, choice in enumerate(choices):
#         print(f"{i}. {choice}")

#     while True:
#         choice_id = input("Enter the ID of your choice (q to quit, enter for default):")
#         if choice_id == "q":
#             print("Goodbye!")
#             sys.exit()
#         elif choice_id == "":
#             print(f"Chose the default value: {default}")
#             return -1

#         try:
#             choice_id = int(choice_id)
#             if choice_id not in range(len(choices)):
#                 print(f"{choice_id} is not within the available range!")
#             else:
#                 print(f"Chose value {choices[choice_id]}")
#                 return choice_id
#         except ValueError:
#             print("Please enter an integer ID!")

# # Function to split configuration keys into common and config-specific keys
# def split_keys(config):
#     common_keys = []
#     config_keys = []
#     for k, v in config.items():
#         if isinstance(v, ConfigBase):
#             config_keys.append(k)
#         else:
#             common_keys.append(k)
#     return config_keys, common_keys

# # Main function for interactive configuration
# def help_config_main():
#     options = {}
#     init_choices = [
#         (PreprocessConfig, "Settings for preprocessing"),
#         (DLTrainerConfig, "Settings for training deep learning models"),
#         (DLTesterConfig, "Settings for testing deep learning models"),
#         (MLTrainerConfig, "Settings for training machine learning models"),
#         (MLTesterConfig, "Settings for testing machine learning models")
#     ]
#     default = DLTrainerConfig()
#     choice_id = query_choices_to_user(
#         [(stringfy(c, drop_suffix=False) + '\t' + desc) for c, desc in init_choices],
#         get_instance_name(default, drop_suffix=False),
#         "Config "
#     )
#     if choice_id == -1:
#         config = default
#     else:
#         config = init_choices[choice_id][0]()

#     help_config_dfs(config, options)
#     config_dict = {"__class__": config.__class__.__name__}
#     config_dict["params"] = options

#     # For debugging purposes
#     # new_config = config_from_json(config_dict)
#     # print(f"Your configuration is as follows:\n {new_config}")

#     # Prompt for saving the configuration to a file
#     path = input("Enter the file name for saving (Default: config.json): ")
#     if not path:
#         path = "config.json"

#     with open(path, "w") as fd:
#         json.dump(config_dict, fd, indent=4)
#         print(f"Your configuration has been written to {path}, "
#               "you can view and modify parameters in this file for future use.")
#     print("Goodbye!")

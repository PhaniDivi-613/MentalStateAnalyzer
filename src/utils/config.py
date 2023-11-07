# Import necessary libraries
from typing import List
import sys
import json
import config as config
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

# Function to get the name of a configuration class without the suffix
def get_instance_name(config, drop_suffix=True):
    name = config.__class__.__name__
    return name[:-6] if drop_suffix else name

import argparse
import json
from utils.config import config_from_json
from dataset.raw import MSARawData
from utils.raw_data import save_raw_data
import os

def create_arg_parser():
    parser = argparse.ArgumentParser()

    # Add the --config-file argument
    parser.add_argument("--config-file", type=str, help="Path to the configuration file")
    parser.add_argument("--save-file", type=str, help="File name to save the preprocessed data", default="msa.joblib")

    return parser


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    current_directory = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    data_directory = os.path.join(parent_directory, "data")
    save_path = os.path.join(data_directory, args.save_file)

    if args.config_file:
        config_file = args.config_file
        with open(config_file) as f:
            json_data = json.load(f)
        configuration = config_from_json(json_data)
        print("Preprocess configuration is as follows:\n {}".format(configuration))
        raw_data = MSARawData(configuration)
        raw_data.describe()
        save_raw_data(raw_data, save_path)
        
    else:
        print("No configuration file provided. Please use --config-file to specify one.")
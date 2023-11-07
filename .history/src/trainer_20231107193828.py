import argparse
import json
from utils.config import config_from_json
from utils.create import create_instance

def create_arg_parser():
    parser = argparse.ArgumentParser()

    # Add the --config-file argument
    parser.add_argument("--config-file", type=str, help="Path to the configuration file")

    return parser


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    if args.config_file:
        config_file = args.config_file
        with open(config_file) as f:
            json_data = json.load(f)
        configuration = config_from_json(json_data)
        print("Trainer configuration is as follows:\n {}".format(configuration))
        print()
        trainer = create_instance(configuration)
        trainer.train()
    else:
        print("No configuration file provided. Please use --config-file to specify one.")
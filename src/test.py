from utils.raw_data import load_raw_data
import os 

current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
data_directory = os.path.join(parent_directory, "data")

raw_data = load_raw_data(os.path.join(data_directory, "msa.joblib"))
print(raw_data)

print(raw_data.label2id)

dic = raw_data.dictionary


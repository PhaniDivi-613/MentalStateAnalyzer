# Mental State Analysis from Twitter Tweets

This repository contains the code and models for analyzing the mental state of Twitter users based on their tweets. We have leveraged Long Short-Term Memory (LSTM) and Memory-Augmented LSTM models for the task, along with static and BERT embeddings, to perform this analysis. By combining natural language processing and machine learning, we aim to provide insights into the emotional and psychological well-being of individuals on Twitter.

## Project Highlights

- **LSTM Models**: We've implemented LSTM models to capture the temporal patterns in users' tweets. These models are capable of learning long-range dependencies in text data, allowing us to better understand how users' mental states evolve over time.

- **Memory Augmented LSTM**: Our use of memory-augmented LSTM enhances the model's ability to store and retrieve relevant information from a user's past tweets, offering a more comprehensive analysis.

- **Embeddings**: We provide two options for embeddings - static and BERT-based embeddings. Static embeddings capture the semantic meaning of words, while BERT embeddings are contextual, offering a richer understanding of the text.

## How to Use This Repository

We perform the tweet classification process in two distinct phases: **Preprocessing** and **Training with Metrics Recording**.
To conveniently control the preprocessing and training, we use JSON files to configure relevant parameters, such as specifying the path to the raw data file during preprocessing, model parameters during training, optimizer parameters, and more. When running, by specifying the configuration file, our code will carry out preprocessing, training work based on the parameters within the file.

### Directory Structure

In the source code, there are six subdirectories and two files, each serving the following purposes:

```bash
├── config          # Includes various parameters and their default settings for preprocessing, and model training.
├── data            # Contains code for data preprocessing and data loading.
├── models          # Primarily includes the implementation of deep learning models.
    ├── __init__.py # Module's initialization file.
├── trainer         # Responsible for model training.
├── utils           # Contains various utility functions.
├── preprocessing.py   # Initial script for the preprocessing step.
├── trainer.py         # Initial script for the Training step.
```
The data folder contains the dataset, processed data, and checkpoints. Examples folder holds a sample configuration json files.

### Preprocessing

In the preprocessing phase, we undertake the following tasks:

1. **Data Preparation**: This involves reading the raw data, tokenizing it, and building a vocabulary to represent the text effectively.

2. **Data Analysis**: We examine label distribution and other data characteristics to gain insights into the dataset.

3. **Data Serialization**: For efficient loading and use in subsequent stages, we save the preprocessed data in a binary format.

```bash
python preprocess.py --config-file ../examples/preprocess-config.json
```
contents of preprocess-config.json

```bash
{
    "__class__": "PreprocessConfig",
    "params": {
        "json_file": "twitter-1h1h.json",
        "tokenizer": "space",
        "nwords": -1,           
        "min_word_count": 1
    }
}
```

Preprocessing will print information about the label distribution for each dataset. Additionally, the processed data is saved in a binary file named `data/msa.joblib`.

### Training with Metrics Recording

In the training phase, our primary focus is on training various models and recording their performance metrics. This phase includes the following steps:

1. **Data Loading**: We load the preprocessed data that we saved during the preprocessing phase.

2. **Model Initialization**: Here, we set up the classification model, configure the optimizer, and define other essential training parameters based on the chosen configuration.

3. **Model Training**: This step involves training the model using the prepared data to obtain the best model that suits the classification task.

4. **Metric Recording**: We record and evaluate the performance metrics of the trained models, allowing for a comparison of their effectiveness.

```bash
python trainer.py --config-file ../examples/trainer-config.json
```
contents of preprocess-config.json

```bash
{
    "__class__": "DLTrainerConfig",
    "params": {
        "score_method": "accuracy",
        "model" : {
            "__class__": "DLModelConfig",
            "params" : {
                "embedding_layer" : {
                    "__class__": "StaticEmbeddingLayerConfig",
                    "params" : {
                        "method" : "random"
                    }
                    
                }
            }
        }
    }
}
```

After training, our code will test the model's performance on the test set and save the best model in the `ckpts` directory.
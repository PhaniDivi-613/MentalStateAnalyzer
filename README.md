![License](https://img.shields.io/apm/l/vim-mode.svg)

Table of Contents:

* [TextClf Introduction](#textclf-introduction)
   * [Preface](#preface)
   * [System Design Approach](#system-design-approach)
   * [Directory Structure](#directory-structure)
* [Installation](#installation)
* [Quick Start](#quick-start)
   * [Preprocessing](#preprocessing)
   * [Training a Logistic Regression Model](#training-a-logistic-regression-model)
   * [Loading a Trained Model for Testing and Analysis](#loading-a-trained-model-for-testing-and-analysis)
   * [Training TextCNN Model](#training-textcnn-model)
* [TODO](#todo)
* [References](#references)

## TextClf Introduction

### Preface

TextClf is a toolbox designed for text classification scenarios. Its goal is to allow users to quickly try various classification algorithm models, adjust parameters, and build baselines through configuration files, so that users can focus more on the characteristics of the data itself and make targeted improvements.

TextClf has the following features:

* It supports both machine learning models such as logistic regression, linear support vector machines, and deep learning models such as TextCNN, TextRNN, TextRCNN, DRNN, DPCNN, Bert, and more.

* It supports various optimization methods, such as `Adam`, `AdamW`, `Adamax`, `RMSprop`, and others.

* It supports various learning rate adjustment methods, such as `ReduceLROnPlateau`, `StepLR`, `MultiStepLR`.

* It supports various loss functions, such as `CrossEntropyLoss`, `CrossEntropyLoss with label smoothing`, `FocalLoss`.

* Users can interact with the program to generate configurations and quickly adjust parameters by modifying the configuration file.

* When training deep learning models, it supports using different learning rates for the `embedding` layer and the `classifier` layer.

* It supports resuming training from a checkpoint.

* It has a clear code structure that allows you to easily add your own models. With `textclf`, you don't need to worry about optimization methods, data loading, and more, so you can focus more on model implementation.

Comparison with other text classification frameworks like [NeuralClassifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier):

* `NeuralClassifier` does not support machine learning models and deep pre-trained models like Bert and Xlnet.

* `TextClf` is more beginner-friendly than `NeuralClassifier`, and its clear code structure makes it easier to extend.

* In particular, for deep learning models, `TextClf` divides them into two parts: the `Embedding` layer and the `Classifier` layer.

  The `Embedding` layer can be randomly initialized word vectors or pre-trained static word vectors (e.g., `word2vec`, `glove`, `fasttext`), or dynamic word vectors like `Bert`, `Xlnet`, and others.

  The `Classifier` layer can be an MLP, CNN, and will also support RCNN, RNN with attention, and various other models in the future.

  By separating the `embedding` layer and `classifier` layer, when configuring deep learning models, we can choose different combinations of `embedding` and `classifier` layers, such as `Bert embedding + CNN`, `word2vec + RCNN`, and so on.

  This way, with relatively little code, `textclf` can cover more possible model combinations.

### System Design Approach

TextClf views the text classification process as consisting of three main stages: **preprocessing, model training, and model testing**.

In the preprocessing stage, the following tasks are primarily performed:

* Reading the raw data, tokenization, and building a vocabulary.
* Analyzing label distribution and other data characteristics.
* Saving the data in a binary format for efficient loading.

Once the data is preprocessed, various models can be trained on it, and their performance can be compared.

The model training stage is responsible for:

* Loading the preprocessed data.
* Initializing the model, optimizer, and other essential training factors based on the configuration.
* Training the model to obtain the best model as needed.

The testing stage's primary functions include:

* Loading the saved model from the training stage for testing.
* Supporting testing through file input or terminal input.

To conveniently control the preprocessing, model training, and model testing stages, `TextClf` uses JSON files to configure relevant parameters, such as specifying the path to the raw data file during preprocessing, model parameters during training, optimizer parameters, and more. When running, by specifying the configuration file, `TextClf` will carry out preprocessing, training, or testing work based on the parameters within the file. For more details, please refer to the [Quick Start](#quick-start) section.


### Directory Structure

In the source code directory of `textclf`, there are six subdirectories and two files, each serving the following purposes:

```bash
├── config      # Includes various parameters and their default settings for preprocessing, model training, and model testing.
├── data        # Contains code for data preprocessing and data loading.
├── models      # Primarily includes the implementation of deep learning models.
├── tester      # Responsible for loading models for testing.
├── __init__.py  # Module's initialization file.
├── main.py     # The interface file for textclf; running textclf calls the main function in this file.
├── trainer     # Responsible for model training.
└── utils       # Contains various utility functions.
```

This directory structure organizes the components and functionality of the `textclf` system.


## Installation

System Requirements: `python >=3.6`

You can install `textclf` using pip as follows:

```bash
pip install textclf
```

Once the installation is successful, you can start using `textclf`!

## Quick Start

Let's take a look at how to use `textclf` to train models for text classification.

In the `examples/toutiao` directory, you will find the following files:

```bash
  3900 lines train.csv
   600 lines valid.csv
   600 lines test.csv
  5100 lines total
```

These data come from the [Toutiao News Classification Dataset](https://github.com/skdjfla/toutiao-text-classfication-dataset) and are used here for demonstration.

The file format is as follows:

```bash
Next Monday (May 7th), those holding these stocks should be cautious   news_finance
How is the immunization plan for pig pseudo-rabies vaccine done?    news_edu
Xiaomi 7 is not here yet! These two Xiaomi phones currently have the highest cost performance, Mi Fans: It's a pity that they can't be bought       news_tech
Any idea relying on technology to solve social justice and fairness is a fantasy        news_tech
Why could Zhuge Liang set fire to Cao Cao's camp with the east wind, but didn't anticipate that it would rain when setting fire to Sima Yi?        news_culture
Several essential travel gadgets with low price, practicality, and high appearance value!  news_travel
How to do annual inspection and purchase insurance for mortgaged cars?    news_car
How much will a house costing 11,000 RMB per square meter sell for in about ten years?      news_house
The first foreigner with Chinese nationality, staying in China for more than fifty years, left behind such words before his death!    news_world
Why do A-share investors lose more when they try to protect their investments?     stock
```

Each line of the file consists of two fields: a sentence and its corresponding label. The sentence and label are separated by the `\t` character.

### Preprocessing

The first step is preprocessing, which involves reading the raw data, tokenization, building a vocabulary, and saving it in a binary format for efficient loading. To control the preprocessing parameters, you need a corresponding configuration file. You can use the `textclf` command's `help-config` feature to quickly generate a configuration file. Run the following command:

```bash
textclf help-config
```

Enter `0` to have the system generate the default `PreprocessConfig` for you, and then save it as `preprocess.json`:

```bash
(textclf) luo@luo-pc:~/projects$ textclf help-config
Config  Options (Default: DLTrainerConfig): 
0. PreprocessConfig     Preprocessing settings
1. DLTrainerConfig      Deep learning model training settings
2. DLTesterConfig       Deep learning model testing settings
3. MLTrainerConfig      Machine learning model training settings
4. MLTesterConfig       Machine learning model testing settings
Enter the ID of your choice (q to quit, enter for default): 0
Chosen value: PreprocessConfig   Preprocessing settings
Enter the filename to save (Default: config.json): preprocess.json
Your configuration has been saved to preprocess.json, where you can view and modify the parameters for future use.
Bye!
```

Open the `preprocess.json` file to see the following content:

```bash
{
    "__class__": "PreprocessConfig",
    "params": {
        "train_file": "train.csv",
        "valid_file": "valid.csv",
        "test_file": "test.csv",
        "datadir": "dataset",
        "tokenizer": "char",
        "nwords": -1,           
        "min_word_count": 1
    }
}
```

The `params` field contains the parameters that can be adjusted. You can find detailed explanations of these fields in the [documentation](docs/preprocess.md). Here, we only need to modify the `datadir` field to the `toutiao` directory (it's better to use an absolute path; if using a relative path, ensure that the current working directory can access that path).

Now, you can perform preprocessing based on the configuration file:

```bash
textclf --config-file preprocess.json preprocess
```

If there are no errors, the output will be as follows:

```bash
(textclf) luo@V_PXLUO-NB2:~/textclf/test$ textclf --config-file config.json preprocess
Tokenize text from /home/luo/textclf/textclf_source/examples/toutiao/train.csv...
3900it [00:00, 311624.35it/s]
Tokenize text from /home/luo/textclf/textclf_source/examples/toutiao/valid.csv...
600it [00:00, 299700.18it/s]
Tokenize text from /home/luo/textclf/textclf_source/examples/toutiao/test.csv...
600it [00:00, 289795.30it/s]
Label Prob:
+--------------------+-------------+-------------+------------+
|                    |   train.csv |   valid.csv |   test.csv |
+====================+=============+=============+============+
| news_finance       |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_edu           |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_tech          |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_culture       |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_travel        |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_car           |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_house         |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_world         |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| stock              |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_story         |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_agriculture   |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_entertainment |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_military      |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_sports        |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| news_game          |      0.0667 |      0.0667 |     0.0667 |
+--------------------+-------------+-------------+------------+
| Sum                |   3900.0000 |    600.0000 |   600.0000 |
+--------------------+-------------+-------------+------------+
Dictionary Size: 2981
Saving data to ./textclf.joblib...
```

Preprocessing will print information about the label distribution for each dataset. Additionally, the processed data is saved in a binary file named `./textclf.joblib`.
(Each category contains the same number of samples.)

For detailed explanations of the preprocessing parameters, please refer to the [documentation](docs/preprocess.md).


### Training a Logistic Regression Model

Similarly, we'll start by generating a configuration file `train_lr.json` using `textclf help-config`. Enter `3` to select the configuration for training a machine learning model. Follow the prompts to choose the `CountVectorizer` (text vectorization method) and the model `LR` (Logistic Regression):

```bash
(textclf) luo@luo-pc:~/projects$ textclf help-config
Config  Options (Default: DLTrainerConfig): 
0. PreprocessConfig     Preprocessing settings
1. DLTrainerConfig      Deep learning model training settings
2. DLTesterConfig       Deep learning model testing settings
3. MLTrainerConfig      Machine learning model training settings
4. MLTesterConfig       Machine learning model testing settings
Enter the ID of your choice (q to quit, enter for default): 3
Chosen value: MLTrainerConfig    Machine learning model training settings
Setting up the vectorizer
Vectorizer Options (Default: CountVectorizer): 
0. CountVectorizer
1. TfidfVectorizer
Enter the ID of your choice (q to quit, enter for default): 0
Chosen value: CountVectorizer
Setting up the model
Model Options (Default: LogisticRegression): 
0. LogisticRegression
1. LinearSVM
Enter the ID of your choice (q to quit, enter for default): 0
Chosen value: LogisticRegression
Enter the filename to save (Default: config.json): train_lr.json
Your configuration has been saved to train_lr.json, where you can view and modify the parameters for future use.
Bye!
```

For more fine-grained configurations, such as logistic regression model parameters or `CountVectorizer` parameters, you can modify them in the generated `train_lr.json` file. Here, we'll use the default configuration for training:

```bash
textclf --config-file train_lr.json train
```

Since the dataset is relatively small, you should see results almost immediately. After training, `textclf` will test the model's performance on the test set and save the model in the `ckpts` directory.

For detailed explanations of the parameters in machine learning model training, please refer to the [documentation](docs/trainer.md).



### Testing and Analyzing a Trained Model

First, use `help-config` to generate the default settings for `MLTesterConfig` and save it in `test_lr.json`:

```bash
(textclf) luo@luo-pc:~/projects$ textclf help-config
Config  Options (Default: DLTrainerConfig): 
0. PreprocessConfig     Preprocessing settings
1. DLTrainerConfig      Deep learning model training settings
2. DLTesterConfig       Deep learning model testing settings
3. MLTrainerConfig      Machine learning model training settings
4. MLTesterConfig       Machine learning model testing settings
Enter the ID of your choice (q to quit, enter for default): 4
Chosen value: MLTesterConfig    Machine learning model testing settings
Enter the filename to save (Default: config.json): test_lr.json
Your configuration has been saved to test_lr.json, where you can view and modify the parameters for future use.
Bye!
```

Modify the `input_file` field in `test_lr.json` to the path of `query_intent_toy_data/test.csv`, and then proceed with testing:

```bash
textclf --config-file test_lr.json test
```

After testing, `textclf` will print the accuracy and the `f1` score for each label:

```bash
Writing predicted labels to predict.csv
Acc in test file: 66.67%
Report:
                    precision    recall  f1-score   support

  news_agriculture     0.6970    0.5750    0.6301        40
          news_car     0.8056    0.7250    0.7632        40
      news_culture     0.7949    0.7750    0.7848        40
          news_edu     0.8421    0.8000    0.8205        40
news_entertainment     0.6000    0.6000    0.6000        40
      news_finance     0.2037    0.2750    0.2340        40
         news_game     0.7111    0.8000    0.7529        40
        news_house     0.7805    0.8000    0.7901        40
     news_military     0.8750    0.7000    0.7778        40
       news_sports     0.7317    0.7500    0.7407        40
        news_story     0.7297    0.6750    0.7013        40
         news_tech     0.6522    0.7500    0.6977        40
       news_travel     0.6410    0.6250    0.6329        40
        news_world     0.6585    0.6750    0.6667        40
             stock     0.5000    0.4750    0.4872        40

          accuracy                         0.6667       600
         macro avg     0.6815    0.6667    0.6720       600
      weighted avg     0.6815    0.6667    0.6720       600
```

For detailed information about the parameters in machine learning model testing, please refer to the [documentation](docs/tester.md).



### Training a TextCNN Model

The process of training a deep learning model like TextCNN is similar to training a logistic regression model. Here's a brief explanation. First, use `help-config` to configure the model. Following the prompts, select `DLTrainerConfig`, then choose `Adam optimizer + ReduceLROnPlateau + StaticEmbeddingLayer + CNNClassifier + CrossEntropyLoss`.

```bash
(textclf) luo@V_PXLUO-NB2:~/textclf/test$ textclf help-config
Config Options (Default: DLTrainerConfig):
0. PreprocessConfig     Preprocessing settings
1. DLTrainerConfig      Deep learning model training settings
2. DLTesterConfig       Deep learning model testing settings
3. MLTrainerConfig      Machine learning model training settings
4. MLTesterConfig       Machine learning model testing settings
Enter the ID of your choice (q to quit, enter for default):
Chosen value: DLTrainerConfig    Deep learning model training settings
Setting optimizer
optimizer Options (Default: Adam):
0. Adam
1. Adadelta
2. Adagrad
3. AdamW
4. Adamax
5. ASGD
6. RMSprop
7. Rprop
8. SGD
Enter the ID of your choice (q to quit, enter for default): 0
Chosen value: Adam
Setting scheduler
scheduler Options (Default: NoneScheduler):
0. NoneScheduler
1. ReduceLROnPlateau
2. StepLR
3. MultiStepLR
Enter the ID of your choice (q to quit, enter for default): 0
Chosen value: NoneScheduler
Setting model
Setting embedding_layer
embedding_layer Options (Default: StaticEmbeddingLayer):
0. StaticEmbeddingLayer
1. BertEmbeddingLayer
Enter the ID of your choice (q to quit, enter for default): 0
Chosen value: StaticEmbeddingLayer
Setting classifier
classifier Options (Default: CNNClassifier):
0. CNNClassifier
1. LinearClassifier
2. RNNClassifier
3. RCNNClassifier
4. DRNNClassifier
5. DPCNNClassifier
Enter the ID of your choice (q to quit, enter for default): 0
Chosen value: CNNClassifier
Setting data_loader
Setting criterion
criterion Options (Default: CrossEntropyLoss):
0. CrossEntropyLoss
1. FocalLoss
Enter the ID of your choice (q to quit, enter for default): 0
Chosen value: CrossEntropyLoss
Enter the filename to save (Default: config.json): train_cnn.json
Your configuration has been saved to train_cnn.json, where you can view and modify the parameters for future use.
Bye!
```

Next, run:

```bash
textclf --config-file train_cnn.json train
```

to start training the TextCNN model with the configured settings.

After training, you can use the `DLTesterConfig` to test the model's performance. If you want to use pretrained static embeddings such as Word2Vec or GloVe, you can easily do so by modifying the configuration file.

If you'd like to try different models like BERT, you can set the `EmbeddingLayer` to `BertEmbeddingLayer` while configuring `DLTrainerConfig` and manually specify the path to the pretrained BERT model in the generated configuration file.

You can find detailed documentation for training deep learning models [here](docs/dl_model.md) and for testing deep learning models [here](docs/tester.md). The [Textclf documentation](docs/README.md) is also available for reference.

## To-Do

- Implement multi-model ensemble evaluation and prediction.
- Load pretrained models and provide API services.
- Auto-tuning (if necessary).

## References

- [DeepText/NeuralClassifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier)
- [pytext](https://github.com/facebookresearch/pytext)

import os
from tabulate import tabulate
from dataset.dictionary import Dictionary, LabelDictionary
from config import PreprocessConfig
from utils.raw_data import (
    tokenize_file,
    create_tokenizer,
    get_label_prob,
    build_label2id
)

class MSARawData(object):
    """Preprocesses data for text classification: tokenization, building a dictionary, and saving it in binary form for easy loading."""

    def __init__(self, config: PreprocessConfig):
        """
        Initialize the data preprocessing with the given configuration.
        
        :param config: Preprocessing settings.
        :type config: PreprocessConfig
        """
        self.config = config
        self.tokenizer = create_tokenizer(config.tokenizer)

        # Tokenize the training, validation, and test data using the specified tokenizer.
        self.pairs = tokenize_file(
            os.path.join(config.datadir, config.json_file),
            self.tokenizer
        )

        # Build a vocabulary dictionary based on the training data.
        self.dictionary = self._build_dictionary()

        # Create a mapping of labels to their corresponding IDs based on the training data.
        self.label2id = build_label2id([labels for _, labels in self.pairs])

    def _build_dictionary(self):
        """
        Build a vocabulary dictionary from the training data.

        :return: A vocabulary dictionary.
        :rtype: Dictionary
        """
        dictionary = Dictionary()
        for texts, _ in self.pairs:
            for text in texts:
                dictionary.add_sentence(text)  # Build the dictionary
        dictionary.finalize(
            nwords=self.config.nwords,
            threshold=self.config.min_word_count
        )
        return dictionary

    def describe(self):
        """
        Output information about the data, including label distributions and dictionary size.
        """
        headers = [
            "",
            self.config.json_file
        ]
        label_prob = get_label_prob([labels for _, labels in self.pairs])
        label_table = []
        for label in label_prob:
            label_table.append([
                label,
                label_prob[label]
            ])
        label_table.append([
            "Sum",
            len(self.pairs),
        ])
        print("Label Probabilities:")
        print(tabulate(label_table, headers, tablefmt="grid", floatfmt=".4f"))

        print(f"Dictionary Size: {len(self.dictionary)}")

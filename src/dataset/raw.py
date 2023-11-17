import os
from tabulate import tabulate
import json
from typing import Iterator
from tqdm import tqdm

from .dictionary import Dictionary, LabelDictionary
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

        # Build a vocabulary dictionary based on the training data.
        self.dictionary, self.labelList = self._build_dictionary_from_file(self.tokenizer, os.path.join(config.datadir, config.json_file))
        self.label2id = build_label2id(self.labelList)
    
    def _build_dictionary_from_file(self, tokenizer, file_path: str, chunk_size: int = 1000):
        """
        Build a vocabulary dictionary from a large file.

        :param file_path: Path to the file containing data.
        :param chunk_size: Size of chunks to read from the file.
        :return: A vocabulary dictionary.
        :rtype: Dictionary
        """
        dictionary = Dictionary()
        labelList = []

        print(file_path)
        with open(file_path, 'r') as file:
            file_lines = file.readlines()
            num_chunks = len(file_lines) // chunk_size
            file_tqdm = tqdm(range(num_chunks), desc="Reading file", unit=" chunks")
            for i in file_tqdm:
                chunk_start = i * chunk_size
                chunk_end = (i + 1) * chunk_size
                chunk = file_lines[chunk_start:chunk_end]
                for line in chunk:
                    data = json.loads(line)
                    if 'posts' in data:
                        for post in data['posts']:
                            if 'text' in post:
                                dictionary.add_sentence(" ".join(tokenizer(post['text'])))
                    if 'label' in data:
                        labelList.append(data['label'])

        dictionary.finalize(
            nwords=self.config.nwords,
            threshold=self.config.min_word_count
        )
        return dictionary, labelList

    def read_file_in_chunks(self, file: list, chunk_size: int) -> Iterator:
        """
        Read a file in chunks.

        :param file: File object as a list of lines.
        :param chunk_size: Size of chunks to read.
        :return: Iterator with file data in chunks.
        """
        for i in range(0, len(file), chunk_size):
            yield file[i:i + chunk_size]


    # def _build_dictionary(self):
    #     """
    #     Build a vocabulary dictionary from the training data.

    #     :return: A vocabulary dictionary.
    #     :rtype: Dictionary
    #     """
    #     dictionary = Dictionary()
    #     for texts, _ in self.pairs:
    #         for text in texts:
    #             dictionary.add_sentence(text)  # Build the dictionary
    #     dictionary.finalize(
    #         nwords=self.config.nwords,
    #         threshold=self.config.min_word_count
    #     )
    #     return dictionary

    def describe(self):
        """
        Output information about the data, including label distributions and dictionary size.
        """
        headers = [
            "",
            self.config.json_file
        ]
        label_prob = get_label_prob(self.labelList)
        label_table = []
        for label in label_prob:
            label_table.append([
                label,
                label_prob[label]
            ])
        label_table.append([
            "Sum",
            len(self.labelList),
        ])
        print("Label Probabilities:")
        print(tabulate(label_table, headers, tablefmt="grid", floatfmt=".4f"))

        print(f"Dictionary Size: {len(self.dictionary)}")

        print(f"Label to ID mapping: {self.label2id}")

    def get_label_2_id(self):
        return self.label2id

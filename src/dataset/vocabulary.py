import torch
from typing import List, Optional
from collections import Counter

class Vocabulary:
    """A mapping from words to unique integer indices"""

    def __init__(
        self,
        padding_symbol="<pad>",
        unknown_symbol="<unk>",
        extra_special_symbols=None,
    ):
        self.unknown_word, self.padding_word = unknown_symbol, padding_symbol
        self.words = []
        self.word_counts = []
        self.word_indices = {}
        self.padding_index = self.add_word(padding_symbol)
        self.unknown_index = self.add_word(unknown_symbol)
        
        if extra_special_symbols:
            for symbol in extra_special_symbols:
                self.add_word(symbol)
        self.num_special_symbols = len(self.words)

    def __eq__(self, other):
        return self.word_indices == other.word_indices

    def __getitem__(self, idx):
        if idx < len(self.words):
            return self.words[idx]
        return self.unknown_word

    def __len__(self):
        """Returns the number of words in the vocabulary"""
        return len(self.words)

    def __contains__(self, word):
        return word in self.word_indices

    def get_index(self, word):
        """Returns the index of the specified word"""
        assert isinstance(word, str)
        if word in self.word_indices:
            return self.word_indices[word]
        return self.unknown_index

    def words_to_string(self, tokens, bpe_symbol=None, escape_unknown=False):
        """Helper for converting a list of word tokens to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tokens) and tokens.dim() == 2:
            return "\n".join(
                self.words_to_string(token_list, bpe_symbol, escape_unknown)
                for token_list in tokens
            )

        def word_string(index):
            if index == self.unknown_index:
                return self.unknown_string(escape_unknown)
            else:
                return self[index]

        if hasattr(self, "begin_of_sentence_index"):
            sentence = " ".join(
                word_string(index)
                for index in tokens
                if (index != self.end_of_sentence_index) and (index != self.begin_of_sentence_index)
            )
        else:
            sentence = " ".join(word_string(index) for index in tokens if index != self.end_of_sentence_index)
        
        return sentence

    def tokens_to_tensor(self, tokens: List[str], max_length: Optional[int] = None):
        """Converts a list of words into a PyTorch tensor of indices.

        Args:
            tokens (List[str]): List of words to be converted to tensor.
            max_length (Optional[int]): If provided, the tensor is truncated or padded to this length.

        Returns:
            torch.Tensor: A tensor containing the indices of the words.
        """
        tensor = torch.ones(len(tokens)).long()
        for i, token in enumerate(tokens):
            tensor[i] = self.get_index(token)

        if max_length is not None:
            if len(tensor) >= max_length:
                tensor = tensor[:max_length]
            else:
                tensor = torch.cat(
                    [tensor, torch.ones(max_length - len(tensor)).long() * self.padding_index]
                )
        return tensor

    def unknown_string(self, escape=False):
        """Return the string for the unknown word, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unknown_word)
        else:
            return self.unknown_word

    def add_word(self, word, count=1):
        """Adds a word to the vocabulary"""
        if word in self.word_indices:
            index = self.word_indices[word]
            self.word_counts[index] = self.word_counts[index] + count
            return index
        else:
            index = len(self.words)
            self.word_indices[word] = index
            self.words.append(word)
            self.word_counts.append(count)
            return index

    def update(self, new_vocabulary):
        """Updates word counts from a new vocabulary"""
        for word in new_vocabulary.words:
            index_in_new = new_vocabulary.word_indices[word]
            if word in self.word_indices:
                index_in_current = self.word_indices[word]
                self.word_counts[index_in_current] = self.word_counts[index_in_current] + new_vocabulary.word_counts[index_in_new]
            else:
                index_in_current = len(self.words)
                self.word_indices[word] = index_in_current
                self.words.append(word)
                self.word_counts.append(new_vocabulary.word_counts[index_in_new])

    def finalize(self, frequency_threshold=-1, word_count_limit=-1):
        """Sorts words by frequency in descending order, ignoring special ones.

        Args:
            - frequency_threshold (int): Minimum word count for a word to be retained.
            - word_count_limit (int): Total number of words in the final vocabulary, including special symbols.
        """
        if word_count_limit <= 0:
            word_count_limit = len(self)

        new_indices = dict(zip(self.words[: self.num_special_symbols], range(self.num_special_symbols)))
        new_words = self.words[: self.num_special_symbols]
        new_word_counts = self.word_counts[: self.num_special_symbols]

        word_count_frequency = Counter(
            dict(
                sorted(zip(self.words[self.num_special_symbols:], self.word_counts[self.num_special_symbols:]))
            )
        )
        
        for word, count in word_count_frequency.most_common(word_count_limit - self.num_special_symbols):
            if count >= frequency_threshold:
                new_indices[word] = len(new_words)
                new_words.append(word)
                new_word_counts.append(count)
            else:
                break

        assert len(new_words) == len(new_indices)

        self.word_counts = list(new_word_counts)
        self.words = list(new_words)
        self.word_indices = new_indices

    def get_padding_index(self):
        """Returns the index of the padding symbol"""
        return self.padding_index

    def get_unknown_index(self):
        """Returns the index of the unknown symbol"""
        return self.unknown_index

    @classmethod
    def load_from_file(cls, file_path):
        """Loads the vocabulary from a text file with the format:

        ```
        <word0> <count0>
        <word1> <count1>
        ...
        ```

        Args:
            file_path (str): Path to the text file containing the vocabulary.

        Returns:
            Vocabulary: An instance of the Vocabulary class.
        """
        vocabulary = cls()
        vocabulary.add_from_file(file_path)
        return vocabulary

    def add_from_file(self, file_path):
        """
        Loads a pre-existing vocabulary from a text file and adds its words
        to this instance.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
            indices_start_line = self._load_meta(lines)
            for line in lines[indices_start_line:]:
                last_space_index = line.rfind(" ")
                if last_space_index == -1:
                    raise ValueError(
                        "Incorrect vocabulary format, expected '<word> <count>'"
                    )
                word = line[:last_space_index]
                count = int(line[last_space_index + 1:])
                self.word_indices[word] = len(self.words)
                self.words.append(word)
                self.word_counts.append(count)

    def save_to_file(self, file_path):
        """Stores the vocabulary into a text file in the format:

        ```
        <word0> <count0>
        <word1> <count1>
        ...
        ```

        Args:
            file_path (str): Path to the text file where the vocabulary will be saved.
        """
        word_count_frequency = Counter(
            dict(
                sorted(zip(self.words[self.num_special_symbols:], self.word_counts[self.num_special_symbols:]))
            )
        )
        
        with open(file_path, 'w') as file:
            for word, count in word_count_frequency.most_common():
                file.write(f"{word} {count}\n")
        print(f"Vocabulary saved to {file_path}!")

    def generate_dummy_sentence(self, length):
        tensor = torch.Tensor(length).uniform_(self.num_special_symbols + 1, len(self)).long()
        tensor[-1] = self.end_of_sentence_index
        return tensor

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_end_of_sentence=True,
        reverse_order=False,
    ):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        num_words = len(words)
        indices = torch.IntTensor(num_words + 1 if append_end_of_sentence else num_words)

        for i, word in enumerate(words):
            if add_if_not_exist:
                word_index = self.add_word(word)
            else:
                word_index = self.get_index(word)
            if consumer is not None:
                consumer(word, word_index)
            indices[i] = word_index
        if append_end_of_sentence:
            indices[num_words] = self.end_of_sentence_index
        return indices

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
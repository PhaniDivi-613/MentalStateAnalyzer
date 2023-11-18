from typing import Tuple, Iterable
from functools import partial
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerBase


from utils.dl_data import texts_to_tensor
from .dictionary import Dictionary

def one_hot_encode(labels, label_to_id):
    num_classes = len(label_to_id)
    one_hot = np.zeros(num_classes)
    for label in labels:
        label_id = label_to_id[label]
        one_hot[label_id] = 1
    return one_hot

class MSADataset2(Dataset):
    def __init__(self, file_path, label_to_id):
        
        self.label_to_id = label_to_id
        self.file_path = file_path
        self.file = open(file_path, 'r')
        self.current_position = 0
        self.num_lines = self.count_lines()

    def __getitem__(self, idx):
        self.file.seek(self.current_position)
        data = self.file.readline().strip()
        if not data:
            self.current_position = 0
            self.file.seek(self.current_position)
            data = self.file.readline().strip()
        self.current_position = self.file.tell()
        sample = json.loads(data)
        lbls = sample["label"]
        sentences = [post["text"] for post in sample["posts"]]
        return sentences, one_hot_encode(lbls, self.label_to_id)

    def __len__(self):
        return self.num_lines-1

    def __del__(self):
        self.file.close()

    def count_lines(self):
        with open(self.file_path, 'r') as file:
            file_lines = file.readlines()
        print(len(file_lines))
        return len(file_lines)

class MSADataset(Dataset):
    def __init__(self, data, label_to_id=None):
        self.data = data
        if label_to_id is None:
            self.label_to_id = {label: id for id, label in enumerate(set(label for _, labels in data for label in labels))}
        else:
            self.label_to_id = label_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentences, labels = self.data[index]
        one_hot_labels = one_hot_encode(labels, self.label_to_id)
        return sentences, one_hot_labels

def build_data_loaders(data, tokenizer_or_dictionary, label2id, config):
    train_size = 0.6
    valid_size = 0.2
    test_size = 0.2

    if data is None:
        train_dataset = MSADataset2(config.train_file, label2id)
        valid_dataset = MSADataset2(config.valid_file, label2id)
        test_dataset = MSADataset2(config.test_file, label2id)
    else:
        msadataset = MSADataset(data, label2id)
        train_dataset, valid_dataset, test_dataset = random_split(msadataset, [int(train_size*len(data)), int(valid_size*len(data)), int(test_size*len(data))])

    collate_fn = get_collate_function(tokenizer_or_dictionary, config.max_len)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last
    )

    valid_data_loader =  DataLoader(
        dataset=valid_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last
    )

    test_data_loader =  DataLoader(
        dataset=test_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last
    )
    
    return train_data_loader, valid_data_loader, test_data_loader
        

def get_collate_function(tokenizer_or_dictionary, max_seq_length):
    if isinstance(tokenizer_or_dictionary, Dictionary):
        collate_fn = partial(collate_batch_with_dictionary, tokenizer_or_dictionary, max_seq_length)
    elif isinstance(tokenizer_or_dictionary, PreTrainedTokenizerBase):
        print("Tokenizer issued")
        collate_fn = partial(collate_batch_with_bert_tokenizer, tokenizer_or_dictionary, max_seq_length)
    return collate_fn

def collate_batch_with_dictionary(dictionary, max_seq_length, data_pairs):
    data_pairs = [(" ; ".join(texts).split()[:max_seq_length], labels) for texts, labels in data_pairs]
    texts, labels = zip(*data_pairs)
    text_lengths = torch.LongTensor([len(text) for text in texts])
    text_tensors = texts_to_tensor(texts, dictionary)

    batch_size = text_tensors.size(0)
    num_padding = max_seq_length - text_tensors.size(1)
    
    if num_padding > 0:
        padding = torch.zeros([batch_size, max_seq_length], dtype=text_tensors.dtype)
        text_tensors = torch.cat([text_tensors, padding], dim=1)
        
    labels = torch.Tensor(labels)
    
    return text_tensors, text_lengths, labels

def collate_batch_with_bert_tokenizer(tokenizer, max_seq_length, data_pairs):
    labels = []
    token_ids = []
    for sentences, lbls in data_pairs:
        tokenized_sentences = [tokenizer(sentence, truncation=True, max_length=10, return_tensors='pt')['input_ids'] for sentence in sentences]
        token_ids.append(torch.cat(tokenized_sentences, dim=1).squeeze())
        labels.append(lbls)

    text_lengths = torch.LongTensor([min(len(text), max_seq_length)  for text in token_ids])
    input_ids = torch.ones(len(token_ids), max_seq_length).long() * tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        input_ids[i][:min(len(tokens), max_seq_length)] = tokens[:min(len(tokens), max_seq_length)]
    labels = torch.Tensor(labels)
    return input_ids, text_lengths, labels

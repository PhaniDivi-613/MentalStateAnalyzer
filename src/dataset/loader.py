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
        data = []
        with open(config.json_file, 'r') as file:
            text_data = file.read()
        text_data = text_data.strip().split('\n')
        json_data = [json.loads(sample) for sample in text_data]
        for sample in json_data:
            lbls = sample["label"]
            sentences = [post["text"] for post in sample["posts"]]
            data.append((sentences, lbls))

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
        tokenized_sentences = [tokenizer(sentence, truncation=True, max_length=100, return_tensors='pt')['input_ids'] for sentence in sentences]
        token_ids.append(torch.cat(tokenized_sentences, dim=1).squeeze())
        labels.append(lbls)

    text_lengths = torch.LongTensor([len(text) for text in token_ids])
    max_text_length = max(text_lengths.max().item(), max_seq_length)
    input_ids = torch.ones(len(token_ids), max_text_length) * tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        input_ids[i][:len(tokens)] = tokens

    labels = torch.Tensor(labels)
    return input_ids, text_lengths, labels

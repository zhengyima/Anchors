import random
from dataclasses import dataclass

import datasets
from typing import Union, List, Tuple, Dict

import torch
from torch.utils.data import Dataset

# from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm

class PassageGenCLSWeightDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, dataset_script_dir, dataset_cache_dir):
        super(PassageGenCLSWeightDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer

        self.passage_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=self._filename,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                'id': datasets.Value("int32"),
                "text": datasets.Value("string"),
                "tokens":[datasets.Value("string")]
            })
        )['train']

        pids_texts = []
        
        self.encoded_dataset = []
        self.total_len = len(self.passage_dataset)    


    
    def __getitem__(self, item):
        passagedata = self.passage_dataset[item]
        # encoded_data = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(passagedata['text']))
        encoded_data = self._tokenizer.convert_tokens_to_ids(passagedata['tokens'])
        encoding = self._tokenizer.encode_plus(
            text=encoded_data,
            max_length=512,
            padding='max_length',
            truncation=True,
        )

        # return encoding
        return {
            "input_ids": np.array(encoding['input_ids']),
            "token_type_ids": np.array(encoding['token_type_ids']),
            "attention_mask": np.array(encoding['attention_mask'])
            }
 
    def __len__(self):
        return self.total_len


class SentenceGenCLSWeightDataset:
    def __init__(self, filename, max_seq_length, tokenizer, dataset_script_dir, dataset_cache_dir):
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer

        self.passage_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=self._filename,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                'sid': datasets.Value("int32"),
                'pno': [datasets.Value("int32")],
                'ano': [datasets.Value("int32")],
                "sentence_tokens":[datasets.Value("string")]
            })
        )['train']

        self.total_len = len(self.passage_dataset)    

    
    def __getitem__(self, item):
        passagedata = self.passage_dataset[item]
        # encoded_data = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(passagedata['text']))
        encoded_data = self._tokenizer.convert_tokens_to_ids(passagedata['sentence_tokens'])
        encoding = self._tokenizer.encode_plus(
            text=encoded_data,
            max_length=512,
            padding='max_length',
            truncation=True,
        )

        # return encoding
        return {
            "input_ids": np.array(encoding['input_ids']),
            "token_type_ids": np.array(encoding['token_type_ids']),
            "attention_mask": np.array(encoding['attention_mask'])
        }
 
    def __len__(self):
        return self.total_len


class SentenceGenAnchorWeightDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, dataset_script_dir, dataset_cache_dir):
        super(SentenceGenAnchorWeightDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer

        self.passage_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=self._filename,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                'sid': datasets.Value("int32"),
                'aid': datasets.Value("int32"),
                'pid': datasets.Value("int32"),
                'aidx': datasets.Value("int32"),
                "tokens":[datasets.Value("string")]
            })
        )['train']

        pids_texts = []
        
        self.encoded_dataset = []
        self.total_len = len(self.passage_dataset)    


    
    def __getitem__(self, item):
        passagedata = self.passage_dataset[item]
        aidx = passagedata['aidx']
        # encoded_data = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(passagedata['text']))
        encoded_data = self._tokenizer.convert_tokens_to_ids(passagedata['tokens'])
        encoding = self._tokenizer.encode_plus(
            text=encoded_data,
            max_length=512,
            padding='max_length',
            truncation=True,
        )
        # print("aidx", aidx)

        # return encoding
        return {
            "input_ids": np.array(encoding['input_ids']),
            "token_type_ids": np.array(encoding['token_type_ids']),
            "attention_mask": np.array(encoding['attention_mask']),
            "aidx": np.array(aidx)
            }
 
    def __len__(self):
        return self.total_len

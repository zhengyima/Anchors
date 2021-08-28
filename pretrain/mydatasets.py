from dataclasses import dataclass
import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
import os
class BERTPretrainedPairWiseDataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer,
            dataset_cache_dir,
            dataset_script_dir,
    ):
        train_file = args.train_file
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        
        print("start loading datasets, train_files: ", train_files)
        mymydatasets = []
        nlp_dataset = datasets.load_dataset(
                f'{dataset_script_dir}/json.py',
                data_files=train_files,
                ignore_verifications=False,
                cache_dir=dataset_cache_dir,
                features=datasets.Features({
                    "pos":{
                        'label': datasets.Value("int32"),
                        "masked_lm_positions": [datasets.Value("int32")],
                        "segment_ids": [datasets.Value("int32")],
                        "tokens_idx":[datasets.Value("int32")],
                        "masked_lm_labels_idxs":[datasets.Value("int32")],
                    },
                    "neg":{
                        'label': datasets.Value("int32"),
                        "masked_lm_positions": [datasets.Value("int32")],
                        "segment_ids": [datasets.Value("int32")],
                        "tokens_idx":[datasets.Value("int32")],
                        "masked_lm_labels_idxs":[datasets.Value("int32")],                    
                    }
                })
            )['train']
        mymydatasets.append(nlp_dataset)


        self.nlp_dataset = nlp_dataset
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)    
        print("loading dataset ok! len of dataset,", self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        pairdata = self.nlp_dataset[item]
        examples = [pairdata['pos'], pairdata['neg']]
        group_batch = []
        for e in examples:
            labels = np.array([-100] * len(e['tokens_idx']))
            masked_lm_positions = e['masked_lm_positions']
            masked_lm_labels = e['masked_lm_labels_idxs']
            labels[masked_lm_positions] = masked_lm_labels

            data = {
                "input_ids": list(e['tokens_idx']),
                "token_type_ids": list(e['segment_ids']),
                "labels": list(labels),
                "next_sentence_label": e['label']
            }
            group_batch.append(BatchEncoding(data))
        return group_batch
@dataclass
class PairCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __call__(
            self, features
    ):
        features_flattened = []
        for f in features:
            features_flattened += [f[0], f[1]]
        features = features_flattened
        batch_size = len(features)
        mlm_labels = []
        for i in range(batch_size):
            mlm_labels.append(features[i]['labels'])
            del features[i]['labels']
        features = super().__call__(features)
        max_len = features['input_ids'].size()[1]
        mlm_labels_matrix = np.ones([batch_size, max_len]) * -100
        for i in range(batch_size):
            mlm_labels_matrix[i][:len(mlm_labels[i])] = mlm_labels[i]
        features['labels'] = torch.LongTensor(mlm_labels_matrix)
        return features

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import torch
from functools import partial
from .utils import sub_label_to_num
import random


class KoBARTSubDataset(Dataset):
    def __init__(self, dataset_path, model_name, model_cls, max_len=512, ignore_index=-100):
        super().__init__()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        if ".csv" in dataset_path:
            self.docs = pd.read_csv(dataset_path)
        elif ".tsv" in dataset_path:
            self.docs = pd.read_csv(dataset_path, sep="\t")
        else:
            raise ValueError
        self.max_len = max_len
        self.model_cls = model_cls
        self.len = self.docs.shape[0]
        self.pad_index = self.tokenizer.pad_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.ignore_index = ignore_index


    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[: self.max_len]

        return inputs

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance["context"])
        input_ids = self.add_padding_data(input_ids)
        
        
        if self.model_cls == 'binary' and instance["subject"] != '여행':
            label = "주거와 생활"
        else:
            label = instance["subject"] 
        label = sub_label_to_num(label)
        
        return {
            "input_ids": np.array(input_ids, dtype=np.int_),
            "labels": torch.tensor(label),
        }

    def __len__(self):
        return self.len


class BlendKoBARTSummaryDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        model_name,
        shuffle_param=2,
        max_shuffle_len=2,
        max_len=512,
        ignore_index=-100,
    ):
        super().__init__()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        if ".csv" in dataset_path:
            self.docs = pd.read_csv(dataset_path)
        elif ".tsv" in dataset_path:
            self.docs = pd.read_csv(dataset_path, sep="\t")
        else:
            raise ValueError
        self.max_len = max_len
        self.len = self.docs.shape[0]
        self.pad_index = self.tokenizer.pad_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.ignore_index = ignore_index
        self.shuffle_param = shuffle_param
        self.max_shuffle_len = max_shuffle_len

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[: self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[: self.max_len]

        return inputs

    def get_processed_item(self, input_ids, label_ids):
        input_ids = self.add_padding_data(input_ids)
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {
            "input_ids": np.array(input_ids, dtype=np.int_),
            "decoder_input_ids": np.array(dec_input_ids, dtype=np.int_),
            "labels": np.array(label_ids, dtype=np.int_),
        }

    def __getitem__(self, idx):
        # original instance
        ori_instance = self.docs.iloc[idx]
        ori_input_ids = self.tokenizer.encode(ori_instance["context"])
        ori_label_ids = self.tokenizer.encode(ori_instance["summary"])
        ori_bos_indices = list(
            filter(lambda x: ori_input_ids[x] == self.bos, range(len(ori_input_ids)))
        )
        # eos/bos token 없을 경우 처리
        if ori_input_ids[-1] != self.eos:
            ori_input_ids.append(self.eos)
        if ori_input_ids[0] != self.bos:
            ori_input_ids.insert(0, self.bos)
        # aux instance
        aux_instance = self.docs.iloc[random.randint(0, len(self.docs) - 1)]
        aux_input_ids = self.tokenizer.encode(aux_instance["context"])
        aux_label_ids = self.tokenizer.encode(aux_instance["summary"])
        aux_bos_indices = list(
            filter(lambda x: aux_input_ids[x] == self.bos, range(len(aux_input_ids)))
        )
        # eos/bos token 없을 경우 처리
        if aux_input_ids[-1] != self.eos:
            aux_input_ids.append(self.eos)
        if aux_input_ids[0] != self.bos:
            aux_input_ids.insert(0, self.bos)
        # aux 첫문장 추출
        aux_first_ids = aux_input_ids[: aux_bos_indices[1]]
        # dual inputs
        dual_input_ids = ori_input_ids + aux_input_ids
        dual_label_ids = ori_label_ids + aux_label_ids
        # aux_first_sentence shuffle은 origin context의 마지막 2 문장에 대해서만 적용.
        dual_noise_input_ids = dual_input_ids.copy()

        dual_insert_bos_idx = random.choice(ori_bos_indices[-2:])
        dual_noise_input_ids = (
            ori_input_ids[:dual_insert_bos_idx]
            + aux_first_ids
            + ori_input_ids[dual_insert_bos_idx:]
        )
        dual_noise_input_ids += aux_input_ids[aux_bos_indices[1] :]

        # dual item
        dual = self.get_processed_item(dual_input_ids, dual_label_ids)
        # dual item + patial shuffle
        dual_noise = self.get_processed_item(dual_noise_input_ids, dual_label_ids)

        result = {}
        result.update({f"{k}": v for k, v in dual.items()})
        result.update({f"noise_{k}": v for k, v in dual_noise.items()})
        return result

    def __len__(self):
        return self.len

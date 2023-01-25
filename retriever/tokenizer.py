import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import re
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

# baseline : https://github.com/boostcampaitech3/level2-mrc-level2-nlp-11

def preprocess(text):
    text = re.sub("\n"," ",text)
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":&<>·\-\'+\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
    return text   

def set_columns(dataset):
    dataset = pd.DataFrame(
        {"context": dataset["context"], "query": dataset["question"], "title": dataset["title"],
        "ground_truth":dataset['ground_truth']
        }
    )

    return dataset


def load_tokenizer(MODEL_NAME):
    special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def tokenize_colbert(dataset, tokenizer ,corpus):

    # for inference
    if corpus == "query":
        preprocessed_data = []
        for query in dataset:
            preprocessed_data.append("[Q] " + preprocess(query))

        tokenized_query = tokenizer(
            preprocessed_data, return_tensors="pt", padding=True, truncation=True, max_length=128,
        )
        return tokenized_query

    elif corpus == "doc":
        preprocessed_data = "[D] " + preprocess(dataset)
        tokenized_context = tokenizer(
            preprocessed_data,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return tokenized_context

    elif corpus == "bm25_hard":
        preprocessed_context = []
        for context in dataset:
            preprocessed_context.append("[D] " + preprocess(context))
        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return tokenized_context
    # for train
    else:
        preprocessed_query = []
        preprocessed_context = []
        for query, context in zip(dataset["query"], dataset["context"]):
            preprocessed_context.append("[D] " + preprocess(context))
            preprocessed_query.append("[Q] " + preprocess(query))
        tokenized_query = tokenizer(
            preprocessed_query, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        tokenized_context = tokenizer(
            preprocessed_context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return tokenized_context, tokenized_query
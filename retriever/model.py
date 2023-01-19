# baseline : https://github.com/boostcampaitech3/level2-mrc-level2-nlp-11

import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


class ColbertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(ColbertModel, self).__init__(config)

        # BertModel 사용
        self.similarity_metric = "cosine"
        self.dim = 128
        self.bert = BertModel(config)
        self.init_weights()
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)

    def forward(self, p_inputs, q_inputs, n_inputs):
        Q = self.query(**q_inputs)
        D = self.doc(**p_inputs)
        if n_inputs:
            N = self.doc(**n_inputs)
            return self.get_score(Q, D, N)
        else:
            return self.get_score(Q, D)

    def query(self, input_ids, attention_mask, token_type_ids):
        Q = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, token_type_ids):
        D = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        D = self.linear(D)
        return torch.nn.functional.normalize(D, p=2, dim=2)

    def get_score(self, Q, D, N=None, eval=False):
        # hard negative N은 train에만 쓰임.
        if eval:
            if self.similarity_metric == "cosine":
                final_score = torch.tensor([])
                for D_batch in tqdm(D):
                    D_batch = np.array(D_batch)
                    D_batch = torch.Tensor(D_batch).squeeze()
                    print(D_batch.shape)
                    p_seqeunce_output = D_batch.transpose(
                        1, 2
                    )  # (batch_size,hidden_size,p_sequence_length)
                    q_sequence_output = Q.view(
                        Q.shape[0], 1, -1, self.dim
                    )  # (batch_size, 1, q_sequence_length, hidden_size)
                    dot_prod = torch.matmul(
                        q_sequence_output, p_seqeunce_output
                    )  # (batch_size,batch_size, q_sequence_length, p_seqence_length)
                    max_dot_prod_score = torch.max(dot_prod, dim=3)[
                        0
                    ]  # (batch_size,batch_size,q_sequnce_length)
                    score = torch.sum(max_dot_prod_score, dim=2)  # (batch_size,batch_size)
                    final_score = torch.cat([final_score, score], dim=1)
                print("final_score!! :",final_score.size())
                return final_score

        else:
            if self.similarity_metric == "cosine":

                p_seqeunce_output = D.transpose(
                    1, 2
                )  # (batch_size, hidden_size, p_sequence_length)
                if N:
                    n_seqeunce_output = N.transpose(1, 2)
                    p_seqeunce_output = torch.cat(
                        [p_seqeunce_output, n_seqeunce_output], dim=0
                    )  # (hard_negative_size, hidden_size, p_sequence_length)
                q_sequence_output = Q.view(
                    Q.shape[0], 1, -1, self.dim
                )  # (batch_size, 1, q_sequence_length, hidden_size)
                dot_prod = torch.matmul(
                    q_sequence_output, p_seqeunce_output
                )  # (batch_size,batch_size, q_sequence_length, p_seqence_length)
                max_dot_prod_score = torch.max(dot_prod, dim=3)[
                    0
                ]  # (batch_size,batch_size,q_sequnce_length)
                final_score = torch.sum(max_dot_prod_score, dim=2)  # (batch_size,batch_size)

                return final_score
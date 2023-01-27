import json
import torch.nn.functional as F
from model import *
from tokenizer import *
import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, set_seed, AutoModel
from pymongo import MongoClient
set_seed(42)


def preprocess(text):
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":&<>·\-\'+\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
    return text  

def tokenize(dataset, tokenizer):
    preprocessed_data="[Q] " + preprocess(query)
    tokenized_query = tokenizer(
        preprocessed_data, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    return tokenized_query

def get_score(Q, D, N=None, eval=False):
        # hard negative N은 train에만 쓰임.
        if eval:
            final_score = torch.tensor([])
            for D_batch in tqdm(D):
                D_batch = np.array(D_batch)
                D_batch = torch.Tensor(D_batch).squeeze()
                #print("Docu_dim_size!! : ",D_batch.shape)
                p_seqeunce_output = D_batch.transpose(
                    1, 2
                )  # (batch_size,hidden_size,p_sequence_length)-> 200 128 512
                #print("Query_dim_size!! : ",Q.size()) 
                q_sequence_output = Q.view(
                    1, 1, -1, 128
                )  # (1, 1, q_sequence_length, hidden_size)
                dot_prod = torch.matmul(
                    q_sequence_output, p_seqeunce_output
                )  # (1,batch_size, q_sequence_length, p_seqence_length)
                max_dot_prod_score = torch.max(dot_prod, dim=3)[
                    0
                ]  # (batch_size,batch_size,q_sequnce_length)
                score = torch.sum(max_dot_prod_score, dim=2)  # (batch_size,batch_size)
                final_score = torch.cat([final_score, score], dim=1)
            print("final_score!! :",final_score.size())
            return final_score

# 몽고 DB 연결하기. -> elastic Search N개 뽑으면 , -> 이거를 받아서 하면댐.
client = MongoClient("mongodb+srv://nlp-08:finalproject@cluster0.rhr2bl2.mongodb.net/?retryWrites=true&w=majority")
db=client['test_database']
blogs=db.blogs
y = blogs.find()
context = list(dict.fromkeys([v["content"]for v in y]))
y = blogs.find()
urls = list(dict.fromkeys([b["url"] for b in y]))

# 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("LeeHJ/colbert_blog")

model = AutoModel.from_pretrained("LeeHJ/colbert_blog")

# 여기서부터 진행.
query = "내일 전라도 여수에 가서 밤바다를 보면서 회를 먹으면 좋을 것 같다. 여수 밤바다 야경이 이쁘다고 한다."

batched_p_embs = []

#블로그 글 임베딩하기
with torch.no_grad():
    print("Start passage embedding......")
    p_embs = []
    for step, p in enumerate(tqdm(context)):
        p = tokenize_colbert(p, tokenizer, corpus="doc").to("cuda")
        p_emb = model.doc(**p).to("cpu").numpy()
        p_embs.append(p_emb)
        if (step + 1) % 200 == 0: # TODO 배치 단위 부분.
            batched_p_embs.append(p_embs)
            p_embs = []
    batched_p_embs.append(p_embs)


# 쿼리 임베딩하기.
with torch.no_grad():
    model.eval()
    q_seqs_val = tokenize(query, tokenizer).to("cuda")
    q_emb = model.query(**q_seqs_val).to("cpu")

# 점수 도출
dot_prod_scores = get_score(q_emb, batched_p_embs, eval=True)

print(dot_prod_scores.size())

rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
print("score !! : ",dot_prod_scores)
print("rank and rank_size!! :",rank)
print(rank.size())

# top k == 5 출력하기.
k = 5

for i in range(k):
    print(f"TOP {i+1} : \n",context[rank[i]])
    print("="*40)

for i in range(k):
    print(f"TOP {i+1} : \n",urls[rank[i]])
    print("="*40)
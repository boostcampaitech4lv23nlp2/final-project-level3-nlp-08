# baseline : https://github.com/boostcampaitech3/level2-mrc-level2-nlp-11


import json
import torch.nn.functional as F
from model import *
from tokenizer import *
import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, set_seed
from datasets import load_from_disk
#from collections import OrderedDict

def main():
    epoch = 10
    MODEL_NAME = "klue/bert-base"
    set_seed(42)
    datasets = load_from_disk("../../data/new_blogs_ict_dataset")
    val_dataset = pd.DataFrame(datasets["validation"])
    
    #val_dataset = val_dataset[:240]

    val_dataset = val_dataset.reset_index(drop=True)
    val_dataset = set_columns(val_dataset)

    tokenizer = load_tokenizer(MODEL_NAME)
    #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(f"./only_blog/only_blog_colbert.pth"))

    print("opening wiki passage...")
    with open("../../data/new_blogs_data.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["context"] for v in wiki.values()]))
    print("wiki loaded!!!")

    query = list(val_dataset["query"])
    ground_truth = list(val_dataset["ground_truth"]) #if using ict_dataset -> #list(val_dataset["ground_truth"])

    batched_p_embs = []
    with torch.no_grad():

        model.eval()

        # 토크나이저
        q_seqs_val = tokenize_colbert(query, tokenizer, corpus="query").to("cuda")
        q_emb = model.query(**q_seqs_val).to("cpu")

        print(q_emb.size())

        print("Start passage embedding......")
        p_embs = []
        for step, p in enumerate(tqdm(context)):
            p = tokenize_colbert(p, tokenizer, corpus="doc").to("cuda")
            p_emb = model.doc(**p).to("cpu").numpy()
            p_embs.append(p_emb)
            if (step + 1) % 50 == 0:
                batched_p_embs.append(p_embs)
                p_embs = []
        batched_p_embs.append(p_embs)
    #torch.save(batched_p_embs,"./embs.pth")

    print("passage tokenizing done!!!!")
    length = len(val_dataset["context"])

    dot_prod_scores = model.get_score(q_emb, batched_p_embs, eval=True)

    print(dot_prod_scores.size())

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    print(dot_prod_scores)
    print(rank)
    print(rank.size())
    torch.save(rank, f"./rank/only_blog{epoch}.pth")

    k = 5
    score = 0

    for idx in range(length):
        print(dot_prod_scores[idx])
        print(rank[idx])
        print()
        for i in range(k):
            if ground_truth[idx] == context[rank[idx][i]]:
                score += 1

    print(f"{score} over {length} context found!!")
    print(f"final score is {score/length}")


if __name__ == "__main__":
    main()
# baseline : https://github.com/boostcampaitech3/level2-mrc-level2-nlp-11

from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from datasets import load_from_disk
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizer import *
from model import *
import json
import pickle

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

BM25_USED = False


def main():

    set_seed(42)

    batch_size = 16
    load_weight_path = False #"./best_model/colbert.pth" #False #"./best_model_aug/colbert_epoch10.pth"
    data_path = "../../data/new_blogs_ict_dataset"

    lr = 5e-5
    args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=6,
        weight_decay=0.01,
        report_to=["wandb"]
    )

    MODEL_NAME = "klue/bert-base"
    tokenizer = load_tokenizer(MODEL_NAME)

    print(data_path)
    datasets = load_from_disk(data_path)
    train_dataset = pd.DataFrame(datasets["train"])
    train_dataset = train_dataset.reset_index(drop=True)

    # bm25 hard negative
    """
    with open("bm25rank_dict.pickle", "rb") as fr:
        bm25rank_dict = pickle.load(fr)

    # hard nagative sample이 담긴 list
    bm25rank_contexts = []
    for idx, row in list(train_dataset.iterrows()):
        bm25rank_contexts.append(bm25rank_dict[row["id"]][0])
    """
    train_dataset = set_columns(train_dataset)

    print("dataset tokenizing.......")
    # 토크나이저
    train_context, train_query = tokenize_colbert(train_dataset, tokenizer ,corpus="both")
    #train_bm25 = tokenize_colbert(bm25rank_contexts, tokenizer, corpus="bm25_hard")

    train_dataset = TensorDataset(
        train_context["input_ids"],
        train_context["attention_mask"],
        train_context["token_type_ids"],
        train_query["input_ids"],
        train_query["attention_mask"],
        train_query["token_type_ids"],
        #train_bm25["input_ids"],
        #train_bm25["attention_mask"],
        #train_bm25["token_type_ids"],
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)

    if load_weight_path:
        model.load_state_dict(torch.load(load_weight_path))
    model.to(device)


    print("model train...")
    trained_model = train(args, train_dataset, model)
    torch.save(trained_model.state_dict(), "only_blog/only_blog_colbert.pth")


def train(args, dataset, model):

    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size
    )

    # Optimizer 세팅
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training 시작
    global_step = 0

    model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        print(f"epoch {epoch}")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss = 0

        for step, batch in enumerate(epoch_iterator):
            model.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }
            
            #n_inputs = {
            #    "input_ids": batch[6],
            #    "attention_mask": batch[7],
            #    "token_type_ids": batch[8],
            #}
            # outputs with similiarity score
            if BM25_USED:
                outputs = model(p_inputs, q_inputs, )#n_inputs
            else:
                outputs = model(p_inputs, q_inputs, None)
            # target: position of positive samples = diagonal element
            targets = torch.arange(0, outputs.shape[0]).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(outputs, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            torch.cuda.empty_cache()
        final_loss = total_loss / len(dataset)
        print("total_loss :", final_loss)
        #torch.save(model.state_dict(), f"./only_blog/compare_colbert_epoch{epoch+1}.pth")

    return model


if __name__ == "__main__":
    main()
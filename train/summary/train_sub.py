import torch
import torch.nn.functional as F
from utils.load_data import KoBARTSubDataset, BlendKoBARTSummaryDataset
from omegaconf import OmegaConf

from utils.trainer import SubTrainer
from transformers import PreTrainedTokenizerFast, AutoConfig, TrainingArguments, Trainer


from models import SubjectModel
from utils.eval_metric import compute_metrics
from utils.callback import SubjectCallback

import argparse


import wandb
import os
import re


def main(cfg):
    # wandb setting
    wandb_cfg = dict()
    for root_key in cfg.keys():
        for key in cfg[root_key].keys():
            wandb_cfg[f"{root_key}.{key}"] = cfg[root_key][key]
    wandb.init(
        project=cfg.exp.project_name,
        name=cfg.exp.exp_name,
        entity="nlp-08-mrc",
        config=wandb_cfg,
    )
    # wandb setting end

    train_data = KoBARTSubDataset(cfg.path.train_path, cfg.model.model_name, cfg.model.cls)
    dev_data = KoBARTSubDataset(cfg.path.dev_path, cfg.model.model_name, cfg.model.cls)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SubjectModel(cfg.model.model_name)
    model.to(device)
    if cfg.model.mode_load_path != 'None':
        model.load_state_dict(
            torch.load(
                cfg.model.mode_load_path
            )
        )
        print(f'loads model\'s param from {cfg.model.mode_load_path}.')

    training_args = TrainingArguments(
        output_dir=f"./results/{cfg.exp.exp_name}",  # output directory
        save_total_limit=cfg.train.save_total_limit,  # number of total save model.
        save_steps=cfg.train.save_steps,  # model saving step.
        num_train_epochs=cfg.train.max_epoch,  # total number of training epochs
        learning_rate=cfg.train.learning_rate,  # learning_rate
        per_device_train_batch_size=cfg.train.batch_size,  # batch size per device during training
        per_device_eval_batch_size=cfg.train.batch_size,  # batch size for evaluation
        warmup_steps=cfg.train.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=cfg.train.weight_decay,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=cfg.train.logging_steps,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        eval_steps=cfg.train.eval_steps,  # evaluation step.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # eval_loss
        greater_is_better=False,
        disable_tqdm=False,
        report_to="wandb",
        remove_unused_columns=False,
    )

    if cfg.trainer.mode == "base":
        print("Use sub trainer")
        trainer = SubTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=dev_data,
            callbacks=[SubjectCallback],
            compute_metrics=compute_metrics,  
        )
    else:
        raise ValueError
    trainer.train()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sub_base")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}.yaml")
    main(cfg)

import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pickle as pickle
import torch
import torch.backends.cudnn as cudnn
import random
from .utils import sub_label_list


def micro_f1(preds, labels):
    label_indices = list(range(len(sub_label_list)))
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def metric_auprc(probs, labels):
    labels = np.eye(len(sub_label_list))[labels]
    score = np.zeros((len(sub_label_list),))

    for c in range(len(sub_label_list)):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    probs, embedding, labels, *_ = pred.predictions
    preds = probs.argmax(-1)  
    
    answer = []
    for i in range(20):
        union = np.union1d(np.where(preds == i), np.where(labels == i))
        answer.append(np.mean(np.where(preds[union] == labels[union], 1, 0)))
    
    f1 = micro_f1(preds, labels)
    auprc = metric_auprc(probs, labels)
    acc = accuracy_score(labels, preds)

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
        'embedding': embedding,
        'labels': labels,
        'answer' : answer,
        'preds' : preds
    }

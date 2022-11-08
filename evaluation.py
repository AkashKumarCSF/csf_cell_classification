from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm

import models
from experiment import ex
from utils import group_probs, map_orig_to_grouped


@ex.capture
def score(y_true: np.array, logits: np.array, classes: List[str], class_groups):
    probs = torch.softmax(torch.Tensor(logits), dim=1).numpy()
    if class_groups is not None:
        class_int_mapping, orig_to_grouped = map_orig_to_grouped(classes,
                                                                 class_groups)

        probs_grouped, y_true = group_probs(y_true, probs, classes,
                                            class_int_mapping,
                                            orig_to_grouped)
        assert np.allclose(np.sum(probs_grouped, axis=1), 1)
        probs = probs_grouped
        classes = class_int_mapping.keys()

        # Make sure the classes are in order
        assert list(class_int_mapping.values()) == list(
            range(1 + max(class_int_mapping.values())))

    labels = range(len(classes))
    y_pred = probs.argmax(axis=1)
    scores = {"acc": accuracy_score(y_true, y_pred),
              "bal_acc": balanced_accuracy_score(y_true, y_pred),
              }
    if probs.shape[1] == 2:
        # Binary classification
        scores["auc"] = roc_auc_score(y_true, probs[:, 1])
    else:
        # Multiclass classification
        scores["auc"] = roc_auc_score(y_true,
                                      probs, average="macro", multi_class="ovo",
                                      labels=labels)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=classes)
    return scores, conf_mat, report


def confmat_df(conf_mat, labels=None):
    """
    Pretty print confusion matrix with row and column sums.

    labels : list of labels or a tuple of (list of labels, list of prediction targets)
    """
    if labels is None:
        labels = range(len(conf_mat))

    if isinstance(labels[0], list):
        assert len(labels) == 2
        labels_annot = labels[0]
        labels_pred = labels[1]
    else:
        labels_annot = labels
        labels_pred = labels
    conf_mat = np.array(conf_mat)
    if conf_mat.shape != (len(labels_annot), len(labels_pred)):
        raise ValueError(f"Confusion matrix as wrong shape: {conf_mat.shape}")
    df = pd.DataFrame(conf_mat, columns=[f"P{i}" for i in labels_pred],
                      index=[f"{i}" for i in labels_annot])
    df["Sum"] = df.sum(axis=1)
    df.loc["Sum"] = df.sum(axis=0)
    return df


@ex.capture
@torch.no_grad()
def evaluate(model, dataloader, criterion, classes, cp_weight, _log,
             return_prediction=False, return_embedding=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    if return_embedding:
        # Can be different for other architectures
        feature_extractor = nn.Sequential(model.network.features, model.network.avgpool, nn.Flatten(1),*list(model.network.classifier.children())[:-1])

    loss = 0.0  # Has to be averaged manually due to possibly non-constant batch size
    y_true = list()
    logits = list()
    paths = list()
    embeddings = list()
    for batch_idx, (data, target, path) in tqdm(enumerate(dataloader),
                                                total=len(dataloader)):
        with torch.cuda.amp.autocast():
            data = data.to(device)
            target = target.to(device)
            if cp_weight is not None and cp_weight > 0:
                output = model.forward_avg(data)
            else:
                output = model(data)
            if return_embedding:
                embedding = feature_extractor(data)  # Only original version
            loss += criterion(output, target).item()
        y_true.append(target.cpu().numpy())
        logits.append(output.cpu().numpy())
        if return_embedding:
            embeddings.append(embedding.cpu().numpy())
        paths.extend(path)
    y_true = np.concatenate(y_true, axis=0)
    logits = np.concatenate(logits, axis=0)
    if return_embedding:
        embeddings = np.concatenate(embeddings, axis=0)
    scores, conf_mat, report = score(y_true.copy(), logits, classes)
    scores["loss"] = loss / len(y_true)
    _log.info('\n' + confmat_df(conf_mat).to_markdown())
    _log.info('\n' + report)
    if return_prediction:
        return scores, y_true, logits, embeddings, paths
    else:
        return scores


@torch.no_grad()
def predict(model, dataloader, device):
    model = model.to(device)
    model.eval()

    y_true = list()
    logits = list()
    paths = list()
    for batch_idx, (data, target, path) in tqdm(enumerate(dataloader),
                                                total=len(dataloader)):
        data = data.to(device)
        target = target.to(device)
        if isinstance(model, models.ConsistencyPriorModel):
            output = model.forward_avg(data)
        else:
            output = model(data)
        y_true.append(target.cpu().numpy())
        logits.append(output.cpu().numpy())
        paths.extend(path)
    y_true = np.concatenate(y_true, axis=0)
    logits = np.concatenate(logits, axis=0)
    scores = None  # TODO
    return scores, y_true, logits, paths

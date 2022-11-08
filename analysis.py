import logging
from pathlib import Path
from shutil import copyfile
from typing import Union, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import torch

import evaluation
import utils


def extract_prob(df: pd.DataFrame):
    """Convert dataframe with class-wise logits to probabilities."""
    return softmax(df.loc[:, df.columns.str.match("prob.*")])


def softmax(logits: pd.DataFrame):
    logits_t = torch.Tensor(logits.values)
    probs = torch.softmax(logits_t, dim=1).numpy()
    logits.loc[:, :] = probs
    return logits


def md_img(descr: str, path: str):
    """Embed image in markdown."""
    # HTML instead of MD's "![{descr}]({path})" due to special chars in paths
    return f'<img src="{path}" alt="{descr}" title="{descr}" />'


def classes_from_cols(df: pd.DataFrame):
    """Extract class names from column names"""
    classes = list(df.columns[df.columns.str.match("prob.*-fold0")].str.replace(
        "prob", "").str.replace("-fold0", ""))
    if len(classes) == 0:
        # Happens when -fold0 is missing from column name
        classes = list(
            df.columns[df.columns.str.match("prob.*")].str.replace(
                "prob", ""))
    return classes


def conf_mat_with_class_groups(y_true, probs_grouped, class_groups, labels=None,
                               normalize=None):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus, in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Read more in the :ref:`User Guide <confusion_matrix>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

        .. versionadded:: 0.18

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and prediced label being j-th class.

    """

    from scipy.sparse import coo_matrix
    from sklearn.utils.multiclass import unique_labels

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    label_to_ind_grouped, orig_to_grouped = utils.map_orig_to_grouped(
        labels, class_groups)

    n_labels = labels.size
    label_to_ind = {y: x for x, y in enumerate(labels)}
    n_labels_grouped = len(label_to_ind_grouped)

    # convert yt, yp into index
    y_pred = probs_grouped.argmax(axis=1)
    y_true = np.array([label_to_ind[x] for x in y_true])

    # # intersect y_pred, y_true with labels, eliminate items not in labels
    # ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    # y_pred = y_pred[ind]
    # y_true = y_true[ind]

    sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    cm = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_labels_grouped), dtype=int,
                    ).toarray()

    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm


def conf_mat_with_subclasses(y_true, y_pred, labels=None, predicted_labels=None,
                             normalize=None):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Read more in the :ref:`User Guide <confusion_matrix>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

        .. versionadded:: 0.18

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and prediced label being j-th class.

    """

    from scipy.sparse import coo_matrix
    from sklearn.utils.multiclass import unique_labels

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    n_labels = labels.size
    label_to_ind = {y: x for x, y in enumerate(labels)}
    label_to_ind_predicted = {y: x for x, y in enumerate(predicted_labels)}
    n_predicted_labels = len(label_to_ind_predicted)

    # convert yt, yp into index
    y_true = np.array([label_to_ind[x] for x in y_true])
    y_pred = np.array([label_to_ind_predicted[x] for x in y_pred])

    # # intersect y_pred, y_true with labels, eliminate items not in labels
    # ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    # y_pred = y_pred[ind]
    # y_true = y_true[ind]

    sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    cm = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_predicted_labels), dtype=int,
                    ).toarray()

    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm


def confusion_table(df: pd.DataFrame, class_groups=None, norm_axis=None) -> \
Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Args:
        norm_axis: Along which access to normalize.
            If None, use absolute scores.
            If 0, normalization would yield the precisions on the diagonal.
            If 1, the recalls.
    """
    y_true = df["label"]
    classes = classes_from_cols(df)
    probs = extract_prob(df).values
    cc = utils.convert_classes

    if class_groups is None:
        y_pred = [classes[c] for c in probs.argmax(axis=1)]
        confmat = skm.confusion_matrix(cc(y_true), cc(y_pred),
                                       labels=cc(classes))

        recalls = confmat.diagonal() / confmat.sum(axis=1)
        precisions = confmat.diagonal() / confmat.sum(axis=0)

        confmat = evaluation.confmat_df(confmat, labels=cc(classes))
    else:
        y_true_orig = y_true.copy()

        if set(y_true) != set(classes):

            # Classes are grouped during training
            tumor_labels = sorted(["Leucaemia", "Lymphoma", "Melanomzellen",
                                   "epitheliale Tumorzelle"])
            labels_orig = [l for l in sorted(set(y_true),
                                             key=lambda x: x.lower()) if
                           l not in tumor_labels] + tumor_labels

            # Here, the labels need to be sorted to be the same as during training
            class_int_mapping, orig_to_grouped = utils.map_orig_to_grouped(
                sorted(labels_orig),
                class_groups)
            classes_grouped = list(class_int_mapping.keys())
            assert set(classes) == set(classes_grouped)

            confmat = conf_mat_with_subclasses(y_true, df["pred-fold0"],
                                               labels_orig, classes)
            y_pred = [classes[i] for i in probs.argmax(axis=1)]
            y_true_grouped = [orig_to_grouped[c] for c in y_true]

            recalls = [
                np.mean(
                    np.array(y_pred)[y_true_orig == c] == orig_to_grouped[c])
                for c in labels_orig]
            recalls = np.array(recalls)
            precisions = skm.precision_score(y_true_grouped, y_pred,
                                             average=None,
                                             labels=list(
                                                 class_int_mapping.keys()))
            confmat = evaluation.confmat_df(confmat,
                                            labels=[cc(labels_orig),
                                                    cc(classes_grouped)])

        else:
            class_int_mapping, orig_to_grouped = utils.map_orig_to_grouped(
                classes,
                class_groups)
            classes_grouped = list(class_int_mapping.keys())

            probs_grouped, y_true = utils.group_probs(y_true, probs, classes,
                                                      class_int_mapping,
                                                      orig_to_grouped)
            y_pred = probs_grouped.argmax(axis=1)
            y_pred = [classes_grouped[y] for y in y_pred]
            y_true = [classes_grouped[y] for y in y_true]

            confmat = conf_mat_with_class_groups(y_true_orig, probs_grouped,
                                                 class_groups,
                                                 labels=classes)

            # Make sure that at least some grouping took place
            assert not np.all(
                np.array(y_true, dtype=str) == y_true_orig.values.astype(str))

            recalls = [
                np.mean(
                    np.array(y_pred)[y_true_orig == c] == orig_to_grouped[c])
                for c in classes]
            recalls = np.array(recalls)
            precisions = skm.precision_score(y_true, y_pred, average=None,
                                             labels=list(
                                                 class_int_mapping.keys()))
            confmat = evaluation.confmat_df(confmat,
                                            labels=[cc(classes),
                                                    cc(classes_grouped)])

    if norm_axis is not None:
        # Cut away the sums
        cm_wo_sum = confmat.iloc[:-1, :-1]
        cm_norm = cm_wo_sum / cm_wo_sum.values.sum(axis=norm_axis,
                                                   keepdims=True)
        confmat.iloc[:-1, :-1] = cm_norm * 100  # In percent

    confmat["recalls"] = recalls.tolist() + [np.mean(recalls)]
    confmat.loc["precisions", :] = precisions.tolist() + [
        np.mean(precisions)] + [0]

    cr = pd.DataFrame(
        skm.classification_report(cc(y_true), cc(y_pred), output_dict=True))

    return confmat, cr


def compute_metrics(df: pd.DataFrame, class_groups=None) -> pd.Series:
    """Compute classification metrics from dataframe."""
    classes, probs, y_pred, y_true = df_to_y(df, class_groups)

    metric_df = pd.Series()
    average = "macro"

    metric_df["recall"] = skm.recall_score(y_true, y_pred, average=average)
    metric_df["precision"] = skm.precision_score(y_true, y_pred,
                                                 average=average)
    metric_df["accuracy"] = skm.accuracy_score(y_true, y_pred)
    metric_df["f1"] = skm.f1_score(y_true, y_pred, average=average)

    y_true_int = [classes.index(yy) for yy in
                  y_true]  # Top k can only handle ordered classes
    classes_int = range(len(classes))
    metric_df["top_2_accuracy"] = skm.top_k_accuracy_score(y_true_int,
                                                           probs,
                                                           k=2,
                                                           labels=classes_int)
    metric_df["top_3_accuracy"] = skm.top_k_accuracy_score(y_true_int,
                                                           probs,
                                                           k=3,
                                                           labels=classes_int)
    metric_df["top_4_accuracy"] = skm.top_k_accuracy_score(y_true_int,
                                                           probs,
                                                           k=4,
                                                           labels=classes_int)

    return metric_df


def df_to_y(df, class_groups=None):
    """Convert dataframe to array of probabilities."""
    y_true = df["label"]
    probs = extract_prob(df).values
    classes = classes_from_cols(df)
    if class_groups is not None:
        class_int_mapping, orig_to_grouped = utils.map_orig_to_grouped(
            classes,
            class_groups)
        probs_grouped, y_true = utils.group_probs(y_true, probs, classes,
                                                  class_int_mapping,
                                                  orig_to_grouped)

        probs = probs_grouped
        classes = list(class_int_mapping.keys())
        y_true = np.array([classes[y] for y in y_true])
    y_pred = probs.argmax(axis=1)
    y_pred = np.array([classes[y] for y in y_pred])
    return classes, probs, y_pred, y_true


def plot(df: pd.DataFrame, class_groups=None, filename=None, title=None):
    classes, probs, y_pred, y_true = df_to_y(df, class_groups)

    fig, ax = plt.subplots(figsize=(8, 8))
    # OVR ROC curve
    for class_id, pos_label in enumerate(classes):
        probs_ = probs[:, class_id]
        try:
            fpr, tpr, thresholds = skm.roc_curve(y_true, probs_,
                                                 pos_label=pos_label)
            auc = skm.roc_auc_score((y_true == pos_label).astype(int),
                                    probs_)
            if auc > 0.999:
                # Don't show "perfect" ROC curves to keep the plot uncluttered.
                raise ValueError("AUC is 1, curve is disregarded.")
        except ValueError as e:
            logging.warning(
                f"Cannot get ROC curve for class {pos_label} due to: {e}")
            continue

        display = skm.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
                                      estimator_name=f"{pos_label}")
        display.plot(ax=ax)
    plt.title(title)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_samples(df: pd.DataFrame, report_root: Path, n=32):
    """For each class, plot samples with highest and lowest loss."""
    cc = utils.convert_classes

    report_root = Path(report_root)
    img_root = report_root / "imgs"
    img_root.mkdir(exist_ok=True)

    prob = extract_prob(df)
    assert prob.values.min() >= 0
    assert prob.values.max() <= 1

    report = list()
    report.append("Legend: predicted class/label")
    classes = classes_from_cols(df)
    for category in classes:
        report.append(f"## {category}")

        labeled_positive = df["label"] == category
        prob_class = prob.loc[labeled_positive, f"prob{category}-fold0"]
        # Find the samples with the lowest probability for the class (high loss)
        high_loss = prob_class.sort_values(ascending=True).head(n=n)
        # Samples with high probability
        low_loss = prob_class.sort_values(ascending=True).tail(n=n)

        for name, samples in [("Highest", high_loss), ("Lowest", low_loss)]:
            report.append(f"### {name} Loss")
            for sample in samples.index:
                dst = img_root / Path(sample).name
                if not dst.exists():
                    copyfile(sample, dst)
                topk = 3
                predicted_classes = prob.loc[sample].nlargest(
                    topk).index.str.replace("prob", "").str.replace("-fold0",
                                                                    "").tolist()
                label = df.loc[sample, "label"]
                report.append(f"{cc(predicted_classes)}/{cc([label])[0]}")
                report.append(md_img(Path(sample).name,
                                     Path(dst).relative_to(report_root)))

    return report


def generate_report(dfs: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    path: Path, title: str, class_groups=None,
                    group_during_training=False,
                    do_plot=False) -> str:
    """
    Generate a evaluation report from a prediction dataframe.

    dfs: Dataframe containing the predictions or a dict that maps a partition to such a dataframe.
    path: Path to store plots etc.
    group_during_training: If true, classes are grouped already during training, so no additional grouping needs to be done here.

    returns: Evaluation report as str in Markdown format
    """
    # Convert single dataframe to list
    if not isinstance(dfs, dict):
        dfs = {"split": dfs}

    report = list()
    report.append(f"# {title}")

    if class_groups is not None:
        report.append(f"Class groups: {class_groups}")

    # *** Metrics ***
    floatfmt = ".0f"
    report.append("## Metrics")
    # Dataframe metric x split
    metric_df = pd.concat(
        [compute_metrics(df,
                         class_groups=None if group_during_training else class_groups)
         for df in dfs.values()],
        axis=1)
    metric_df.columns = dfs.keys()
    report.append((100 * metric_df).to_markdown(floatfmt=floatfmt))
    report.append("\n")

    img_path = path / f"imgs.{title}"
    img_path.mkdir(exist_ok=True)
    # *** Confusion matrix ***
    report.append("## Confusion Matrix")
    for split, df in dfs.items():
        report.append(f"### {split}")
        confmat, cr = confusion_table(df,
                                      class_groups=None if group_during_training else class_groups)

        # Convert precision/recall to percent
        confmat["recalls"] *= 100
        confmat.loc["precisions"] *= 100
        cr.loc[["precision", "recall", "f1-score"]] *= 100
        cr.loc["support", "accuracy"] *= 100

        report.append(confmat.to_markdown(floatfmt=floatfmt))
        report.append("\n")
        report.append(cr.to_markdown(floatfmt=floatfmt))
        report.append("\n")

    if do_plot:
        report.append("## Plots")
        for split, df in dfs.items():
            plot_file = img_path / f"roc_{split}.png"
            plot(df, filename=plot_file, title=split,
                 class_groups=None if group_during_training else class_groups)
            report.append(
                md_img(f"ROC {split}", f"{plot_file.relative_to(path)}"))

    if "test" in dfs:
        report.append("# Test Samples with Extreme Loss")
        report.extend(plot_samples(dfs["test"], path))

    report = "\n".join(report)

    return report

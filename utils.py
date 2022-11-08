import json
from pathlib import Path
from typing import List

import matplotlib.cm
import numpy as np
import pandas as pd
import requests
from PIL import Image

import local_config


def setup_prediction_df(y_score: np.ndarray, y_true: np.ndarray, paths: list,
                        classes: list, classes_true=None,
                        suffix="") -> pd.DataFrame:
    """
    Combine predicted probabilities and labels to one dataframe.
    y_score: `(num_samples, num_classes)` array of class probabilities
    y_true: Array of length `num_samples` with entries from 0 to `num_classes`
    paths: List of length `num_samples` with paths to the images. Used as index.
    classes: List of length `num_classes` with class names as strings
    suffix: Optional string to append to the column names, e.g. fold indices.
    """
    assert y_score.ndim == 2, y_score.ndim
    y_pred = np.argmax(y_score, axis=1)
    if classes_true is None:
        classes_true = classes
    prediction_df = pd.DataFrame(
        dict(
            **{f"prob{classes[class_id]}{suffix}": y_score_class for
               class_id, y_score_class in enumerate(y_score.T)},
            **{f"pred{suffix}": [classes[y] for y in y_pred]},
            label=[classes_true[y] for y in y_true]),
        index=paths
    )

    return prediction_df


def send_to_slack(message):
    try:

        with open(Path.home() / "slack.json", "r") as f:
            slack_config = json.load(f)
            webhook_url = slack_config["webhook_url"]

        data = {
            "username": "snakebot",
            "icon_emoji": ":snake:",
            "text": message,
        }
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        requests.post(webhook_url, data=json.dumps(data), headers=headers)
    except Exception as e:
        print(f"Cannot send message to slack because: {e}")


def group_probs(y_true, probs, classes, class_int_mapping, orig_to_grouped):
    # y_true should be castable to int, otherwise convert it
    try:
        y_true = np.array(y_true).astype(int)
    except ValueError:
        y_true = np.array([classes.index(c) for c in y_true])

    classes_grouped = class_int_mapping.keys()
    # Make sure the classes are in order
    assert list(class_int_mapping.values()) == list(
        range(1 + max(class_int_mapping.values())))
    probs_grouped = np.zeros((len(probs), len(classes_grouped)))
    for c in classes:
        # Group labels
        c_idx = classes.index(c)
        y_true[y_true == c_idx] = class_int_mapping[orig_to_grouped[c]]

        # Sum probabilities
        grouped_c_idx = class_int_mapping[orig_to_grouped[c]]
        probs_grouped[:, grouped_c_idx] += probs[:, c_idx]
    assert np.allclose(probs_grouped.sum(axis=1), 1)
    return probs_grouped, y_true


def map_orig_to_grouped(classes, class_groups):
    """
    Args:
        classes: list of classes (long format)
        class_groups: mapping from class_group to class (long format)
    Returns:
        class_int_mapping: Maps original class name to a class group ID in [0, num_groups)
        orig_to_groups: Maps original class name to class group name.
    """
    class_int_mapping = dict()
    orig_to_grouped = dict()
    class_ctr = 0
    for orig_class in classes:
        # Get the indices associated to the group
        for class_group_name, class_group in class_groups.items():
            if orig_class in class_group:
                if class_group_name not in class_int_mapping:
                    class_int_mapping[class_group_name] = class_ctr
                    class_ctr += 1
                break
        else:
            class_int_mapping[orig_class] = class_ctr
            class_ctr += 1
            class_group_name = orig_class

        orig_to_grouped[orig_class] = class_group_name
    return class_int_mapping, orig_to_grouped


def convert_classes(classes: List[str]) -> List[str]:
    """Map class names to 4 chars."""
    class_mapping = {"aktivierter Lymphozyt": "akLy",
                     "aktivierter Monozyt": "akMo",
                     "Lymphozyt": "Lyzy",
                     "Erythrophage": "Erph",
                     "Erythrozyt": "Erzy",
                     "Lymphoma": "Lyma",
                     "MonoAct": "MoAc",
                     }

    classes_abbr = [class_mapping[c] if c in class_mapping else c[:4] for c in
                    classes]

    # Make sure that the class count is the same
    assert len(set(classes)) == len(set(classes_abbr)), (
        (len(set(classes)), len(set(classes_abbr))),
        (set(classes), set(classes_abbr)))

    return classes_abbr


def hm_to_image(hm, q=95) -> Image:
    scaling_method = "robust_lrp"
    if scaling_method == "robust_lrp":
        a_max = np.percentile(hm[hm > 0], q)
        hm_clipped = np.clip(hm, a_min=-a_max, a_max=a_max)
    else:
        a_max = np.abs(hm).max()
        hm_clipped = hm
    # hm = skimage.exposure.rescale_intensity(hm,
    #                                         out_range=(
    #                                         0,
    #                                         255)).astype(
    #     np.uint8)
    hm = (255 * (hm_clipped / a_max + 1) / 2).astype(
        np.uint8)
    cm = matplotlib.cm.get_cmap("coolwarm")
    hm = cm(hm)
    hm = Image.fromarray(
        (255 * hm[:, :, :3]).astype(np.uint8))
    return hm


def load_predictions(run_id):
    log_dir_root = local_config.log_dir_root
    log_dir_base = Path(log_dir_root) / str(run_id)
    log_dir = Path(log_dir_base)
    predictions_test = pd.read_csv(log_dir / f"predictions_test.csv",
                                   index_col=0)
    predictions_test["label_orig"] = predictions_test.index.map(
        lambda p: Path(p).parent.name)
    predictions_test["idx"] = range(len(predictions_test))
    return log_dir, predictions_test


def load_df(run_id, stub):
    df = pd.read_csv(stub.format(run_id), index_col=0)
    df["label_orig"] = df.index.map(lambda p: Path(p).parent.name)
    return df

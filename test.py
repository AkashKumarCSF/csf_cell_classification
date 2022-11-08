import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import analysis
import data_splitting
import dataloading
import utils
from evaluation import score
from experiment import get_dataset, setup_model
from utils import group_probs, map_orig_to_grouped


@pytest.fixture
def file_stats():
    return "test_data/statistics_all.csv"


@pytest.fixture
def example_df():
    return pd.read_csv("test_data/predictions_xval.csv", index_col=0)


@pytest.fixture
def datafolder_random():
    classes = list("ABCDEF")
    for c in classes:
        (Path("test_data") / "imgfolder" / c).mkdir()
        for i in range(np.random.randint(10, 50)):
            (Path("test_data") / "imgfolder" / c / f"i.jpg").touch()

    return Path("test_data") / "imgfolder"


def test_prediction_df():
    N = 10
    K = 3
    inner_fold_idx = 1
    y_score = np.random.uniform(0, 1, size=(N, K))
    y_true = np.random.choice(range(K), size=(N,))
    paths = np.random.choice(list("xvlcwkhguiaeosnr"), size=(N,))
    classes = np.random.choice(list("xvlcwkhguiaeosnr"), size=(K,))

    df = utils.setup_prediction_df(y_score, y_true, paths, classes,
                                   suffix=f"-fold{inner_fold_idx}")
    print()
    print(df)


def test_load_stats(file_stats):
    df = data_splitting.load_stats(file_stats)

    print(df.head())
    expected_classes = {'Leucaemia', 'Mitose', 'aktivierter Lymphozyt',
                        'aktivierter Monozyt',
                        'Plasmazelle', 'eosinophiler Granulozyt',
                        'Hämosiderophage',
                        'Erythrozyt', 'Makrophagen', 'Monozyt',
                        'neutrophiler Granulozyt',
                        'Lymphozyt', 'Autolytische Zelle', 'Kernschatten',
                        'azelluläre (Färbe)artefakte', 'Hämatoidin',
                        'Proteinpräzipitat',
                        'Erythrophage', 'Hautschuppe', 'epitheliale Tumorzelle',
                        'Artifizielle Zelle', 'Lymphoma', 'Melanomzellen',
                        'hämatopoetische Tumorzelle', 'Bakterien'}

    assert set(df.columns) == expected_classes


def test_random_split_grouped(file_stats):
    file_best_indices = "test_data/out/split_file.json"
    file_stats = "test_data/statistics.csv"
    num_trials = "100"
    try:
        data_splitting.random_split_grouped(
            [file_stats, file_best_indices, num_trials, f"--file_indices={file_best_indices}"])
    except SystemExit:
        # Click always returns a SystemExit, so we need to catch it
        pass


@pytest.mark.parametrize("classes,class_groups", [
    (list("abcdef"), None),
    (list("abcdef"), {"x": ["b", "d"]}),
    (list("ab"), None),  # Binary classification
])
def test_score(classes, class_groups):
    k = len(classes)
    n = 100
    y_true = np.random.randint(k, size=n)
    logits = np.random.normal(size=(n, k))
    scores, conf_mat, report = score(y_true, logits, classes, class_groups)
    if class_groups is None:
        assert conf_mat.shape == (k, k)
    else:
        assert conf_mat.shape == (k - 1, k - 1)


class TestAnalysis:
    @pytest.mark.parametrize("class_groups", [
        None,
        {"Tumor": ["epit", "Mela"]}
    ])
    @pytest.mark.parametrize("group_during_training", [True, False])
    def test_confusion_table(self, example_df, class_groups,
                             group_during_training):
        conf, cr = analysis.confusion_table(example_df,
                                            class_groups=class_groups)

        print("\n", conf.to_markdown())

        print("\n", cr.to_markdown())

        assert "recalls" in conf.columns
        assert "precisions" in conf.index
        assert "Sum" in conf.columns
        assert "Sum" in conf.index

    @pytest.mark.parametrize("class_groups", [
        None,
        {"Tumor": ["epit", "Mela"]}
    ])
    def test_compute_metrics(self, example_df, class_groups):
        metric_df = analysis.compute_metrics(example_df, class_groups)
        print("\n", metric_df.to_markdown())

    def test_plot(self, example_df):
        filename = "test_data/out/roc.png"
        analysis.plot(example_df, filename=filename)

    @pytest.mark.xfail(reason="Images are not available locally.")
    def test_plot_samples(self, example_df):
        analysis.plot_samples(example_df, "")

    def test_report(self, example_df):
        report = analysis.generate_report(example_df, Path("test_data"),
                                          "Xval Report")
        with open("test_data/out/test_report.md", "w") as f:
            f.write(report)

    @pytest.mark.xfail(reason="Images are not available locally.")
    def test_report_combine(self, example_df):
        dfs = {k: example_df for k in ["train", "xval", "test"]}
        report = analysis.generate_report(dfs, Path("test_data"),
                                          "Combined Report")
        with open("test_data/out/test_report_combined.md", "w") as f:
            f.write(report)

    def test_group_probs(self, example_df):
        class_groups = {"Tumor": ["epit", "Mela"]}
        classes = analysis.classes_from_cols(example_df)  # Output of the model
        probs = analysis.extract_prob(example_df).values  # N, K
        assert len(classes) == probs.shape[1]
        y_true = example_df["label"]

        class_int_mapping, orig_to_grouped = map_orig_to_grouped(
            classes,
            class_groups)
        classes_grouped = list(class_int_mapping.keys())

        # Assert that the two original classes are gone but the new group class was added
        assert (set(classes) - set(class_groups["Tumor"])) | set(
            ["Tumor"]) == set(class_int_mapping.keys())

        assert set(orig_to_grouped.keys()) == set(classes)
        assert set(orig_to_grouped.values()) == set(class_int_mapping.keys())

        y_true_orig = y_true.copy()
        probs_grouped, y_true = group_probs(y_true, probs, classes,
                                            class_int_mapping, orig_to_grouped)
        y_true = np.array([classes_grouped[y] for y in y_true])

        assert set(y_true_orig[y_true != y_true_orig]) == set(
            class_groups["Tumor"])

        assert set(y_true_orig) == set(classes)
        assert set(y_true) == set(classes_grouped)

        # Assert that probs stay the same for all ungrouped classes
        for c in set(classes).intersection(set(classes_grouped)):
            assert np.allclose(probs[:, classes.index(c)],
                               probs_grouped[:, classes_grouped.index(c)])


class TestDataloading:
    @pytest.mark.parametrize("class_groups", [
        {"Tumor": ["epit", "Lyma"]},
        None,
    ])
    @pytest.mark.parametrize("max_num", [None, 3])
    @pytest.mark.parametrize("classes_to_keep", [None])  # TODO
    def test_dataset(self, class_groups, max_num, classes_to_keep):
        data_root = "test_data/data_root"
        group_samples = "test_data/supplementary_table.ods"
        group_during_training = True
        dataset = get_dataset(data_root, group_samples, max_num, class_groups,
                              group_during_training, classes_to_keep)

        # find test_data/data_root -name "*.png" | wc -l => 20
        n_expected = 20
        num_classes_orig = 4  # ls test_data/data_root | wc -l => 4
        if max_num is None:
            assert len(dataset) == n_expected
        else:
            # In every class, there are 5 > 3 samples, so it should be equal
            len_new = len(dataset.subsample(range(len(dataset)), max_num))
            assert len_new == num_classes_orig * max_num
        targets = [d[1] for d in dataset]

        num_classes_expected = 4 if class_groups is None else 3
        assert len(set(targets)) == num_classes_expected

    def test_subsampling(self):
        class DummyDataset(dataloading.GroupedImageFolder):
            def __init__(self, num_samples, num_groups, num_classes):
                self.targets = np.random.randint(num_classes, size=num_samples)
                self.imgs = list(zip(range(num_samples),
                                     self.targets))
                self.groups = np.random.randint(num_groups, size=num_samples)
                self.samples = self.imgs
                self.class_to_idx = {c: c for c in range(num_classes)}
                self.loader = lambda x: x
                self.transform = None
                self.target_transform = None
                self.classes = list(range(num_classes))

        num_samples = 100
        num_groups = 4
        num_classes = 5
        max_num = 3
        dataset = DummyDataset(num_samples, num_groups, num_classes)
        num_train_samples = int(0.8 * num_samples)
        indices = np.random.permutation(np.arange(num_samples))[
                  :num_train_samples]
        train_dataset = dataloading.SubsetWithTransform(dataset, indices)
        indices_sub = dataset.subsample(indices, max_num)
        train_dataset_sub = dataloading.SubsetWithTransform(dataset,
                                                            indices_sub)

        assert len(indices_sub) == max_num * num_classes
        # Do we have the correct number of train samples?
        assert dataloading.class_distribution(dataset, dataset.classes)[
                   "freq"].sum() == num_samples
        assert dataloading.class_distribution(train_dataset, dataset.classes)[
                   "freq"].sum() == num_train_samples

        # Make sure that we have at least max_num samples per class
        assert dataloading.class_distribution(train_dataset, dataset.classes)[
                   "freq"].min() >= max_num

        assert (dataloading.class_distribution(train_dataset_sub,
                                               dataset.classes)[
                    "freq"] == max_num).all()


class TestModels:
    @pytest.mark.parametrize("model_name",
                             ["vgg16", "resnet50", "efficientnetb4",
                              "efficientnetb0", "densenet121"])
    def test_setup_model(self, model_name):
        num_classes = 2
        m = setup_model(model_name, False, None, num_classes, 0)

        x = torch.zeros((4, 3, 128, 128))
        y = m(x)
        assert y.shape[1] == num_classes


def create_dummy_split_file():
    """Used to split the test data in test_data/data_root"""
    outfile = "test_data/split_file.json"
    num_classes = 4
    samples_per_class = 5
    split = {
        "train": list(range(0, samples_per_class * num_classes, samples_per_class)),
        "val": list(range(1, samples_per_class * num_classes, samples_per_class)),
        "test": list(range(2, samples_per_class * num_classes, samples_per_class)),
    }
    with open(outfile, "w") as f:
        json.dump(split, f)


if __name__ == '__main__':
    create_dummy_split_file()
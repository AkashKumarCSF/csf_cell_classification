import json
import argparse
import click
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm


def jensen_shannon_divergence(ps):
    """
    Mean KL-divergence of the probability vectors to their mean.

    ps : List of probability vectors.
    """

    # If not all classes have samples, it's already invalid
    if not np.all(ps > 0):
        return np.nan

    def kl(p, q):
        return np.sum(p * np.log(p / q))

    m = np.mean(ps, axis=1)

    return np.mean([kl(p, m) for p in ps.T])


def sample(df: pd.DataFrame):
    """
    Randomly split the samples in the dataframe into train, val and test.
    """

    indices = {k: list() for k in ["test", "val", "train"]}

    # Optionally, first distribute challenging, manually selected classes without balancing
    # These classes should be non-overlapping, i.e. any case can have samples of only one of these classes
    manual_classes = ["Leucaemia", "Lymphoma", "Melanomzellen",
                      "epitheliale Tumorzelle"]
    #manual_classes = []
    df_dropped = df.copy()
    #print("columns: ", df_dropped.columns)

    for c in manual_classes:
        if c in df_dropped.columns:
            # Get cases belonging to that label
            cases = list(df_dropped.index[df_dropped[c] > 0])
            np.random.shuffle(cases)
            n = len(cases)
            num_test = int(np.ceil(0.2 * n))
            num_val = int(np.ceil(0.3 * (n - num_test)))
            indices["test"].extend(cases[:num_test])
            indices["val"].extend(cases[num_test:num_test + num_val])
            indices["train"].extend(cases[num_test + num_val:])

            df_dropped.drop(index=cases, errors="ignore", inplace=True)

    # Assert that no samples of the manual classes remain in the df
    relevant_cols = list(set(manual_classes).intersection(df.columns))
    if relevant_cols:
        assert df_dropped[
                   set(manual_classes).intersection(df.columns)].values.max() == 0
        # Assert that at least some samples were dropped
        assert len(df_dropped) < len(df)

    indices["test"].extend(np.random.choice(df_dropped.index,
                                            size=np.random.randint(5, 8),
                                            replace=False))
    # Remove already assigned samples. Allow errors because some samples were dropped already before.
    df_tmp = df_dropped.drop(index=indices["test"], errors="ignore")
    indices["val"].extend(np.random.choice(df_tmp.index,
                                           size=np.random.randint(2, 7),
                                           replace=False))
    df_tmp = df_tmp.drop(index=indices["val"], errors="ignore")
    indices["train"].extend(df_tmp.index)

    # Check if partitioning is valid
    # Check that the partitions are disjoint
    check_partitioning(df, indices)

    splits = ["train", "val", "test"]
    num_samples = pd.DataFrame(
        {s: sum([df.loc[idx] for idx in indices[s]]) for s in splits})
    for s in splits:
        num_samples[s] = num_samples[s] / num_samples[s].sum()

    #print("num_samples: {}".format(num_samples))
    missing_classes = num_samples[(num_samples == 0).any(axis=1)]

    #print("\n✅ Class presence in each split:")
    for split in splits:
        present_classes = df.loc[indices[split]].sum(axis=0)
        present_classes = present_classes[present_classes > 0].index.tolist()
        #print(f"{split.capitalize()} split: {present_classes}")

    #if not missing_classes.empty:
    #    print("⚠️ Classes with 0 samples in at least one split:")
    #    print(missing_classes)

    jsd = jensen_shannon_divergence(num_samples.values)
    return jsd, indices, num_samples


def check_partitioning(df, indices):
    """Assert that the partitioning is valid, i.e. each sample is mapped to only one partition, and the partitions cover the entire dataset."""
    assert len(
        set(indices["train"]).intersection(set(indices["test"]))) == 0
    assert len(set(indices["train"]).intersection(set(indices["val"]))) == 0
    assert len(set(indices["val"]).intersection(set(indices["test"]))) == 0
    # Check that the partitions cover all indices
    assert set(indices["train"]).union(set(indices["val"])).union(
        set(indices["test"])) == set(df.index)


def load_stats(file_stats: str):
    """
    Load dataframe with class frequencies per case.

    file_stats: CSV file with class frequencies.
    """
    df = pd.read_csv(file_stats).set_index('Case')
    df = df.drop("total", axis=0)
    df = df.drop("total", axis=1)
    return df


@click.command()
@click.argument("file_stats", type=str)
@click.argument("file_best_indices", type=str)
@click.argument("num_trials", type=int)
@click.option("--file_indices", type=str, default=None)
@click.option("--plot_file", type=str, default=None)
def random_split_grouped(file_stats: str, file_best_indices: str,
                         num_trials: int,
                         file_indices: str = None, plot_file: str = None):
    """
    Split grouped by cases such that each class is present in each split and the
    class distribution in the splits are similar.
    The data are split randomly multiple times and the most balanced splitting
    is selected. Here, class balance is quantified by the Jensen Shannon
    divergence of the class distributions.

    file_stats : CSV file with the class frequencies of the samples.
    file_best_indices : The found splitting is saved to this file (JSON format).
    num_trials : The number of trials, i.e. random splits of the data.
    file_indices: All valid partitions are stored to this file long with their JSD (YAML format).
    plot_file : If not None, plot the class histogram to this file.
    """
    print("file_stats: ", file_stats)
    print("file_best_indices: ", file_best_indices)
    print("num_trials: ", num_trials)
    print("file_indices: ", file_indices)
    print("plot_file: ", plot_file)

    print("Loading data...")
    df = load_stats(file_stats)

    #df = load_stats(file_stats)
    best_indices = None
    best_jsd = np.inf

    # Store the results of all valid partitions
    valid_indices = list()
    for i in tqdm(range(int(num_trials))):
        jsd, indices, num_samples = sample(df.copy())
        #print("jsd: ", jsd)

        if not np.isnan(jsd):
            print("jsd: ", jsd)
            tqdm.write(f"{jsd:.5f} (current best: {best_jsd:.5f})")
            valid_indices.append({**indices, "jsd": str(jsd), "iteration": i})
            if jsd < best_jsd:
                best_jsd = jsd
                best_indices = indices

                if plot_file is not None:
                    fig, ax = plt.subplots(figsize=(12, 12))
                    num_samples.plot.bar(sharex=True, ax=ax)
                    ax.set_title(jsd)
                    ax.set_yscale("log")
                    fig.savefig(plot_file)

    with open(file_indices, "w") as f:
        yaml.dump(valid_indices, f)
    if best_indices is None:
        raise Exception(
            f"Could not find a valid data partitioning after {num_trials} trials.")
    else:
        with open(file_best_indices, "w") as f:
            output_dict = {k: list(v) for k, v in best_indices.items()}
            output_dict["jsd"] = best_jsd
            output_dict["num_trials"] = num_trials
            json.dump(output_dict, f, indent=4)


if __name__ == '__main__':
    random_split_grouped()

import json
import sys
from pathlib import Path

import pandas as pd

import analysis
import local_config

fs_observer_dir = local_config.log_dir_root

if __name__ == '__main__':

    print("Evaluation")

    try:
        smk_to_sacred_id = snakemake.input[1]
    except NameError:
        # For debugging, script can be called without snakemake
        smk_to_sacred_id = sys.argv[2]
    with open(smk_to_sacred_id, "r") as f:
        sacred_id = f.read().strip()

    run_dir = Path(fs_observer_dir) / sacred_id
    splits = ["train", "xval", "test"]
    dfs = {split: pd.read_csv(run_dir / f"predictions_{split}.csv", index_col=0)
           for split in splits}

    with open(Path(fs_observer_dir) / sacred_id / "config.json") as f:
        config = json.load(f)
        class_groups = config.get("class_groups", None)
        group_during_training = config.get("group_during_training", False)

    # Generate markdown reports with and without class grouping
    title = f"Report {sacred_id}"
    report = analysis.generate_report(dfs, run_dir, title)
    with open(run_dir / "report.md", "w") as f:
        f.write(report)

    if class_groups is not None:
        title = f"Report Grouped {sacred_id}"
        report = analysis.generate_report(dfs, run_dir, title,
                                          class_groups=class_groups,
                                          group_during_training=group_during_training)

        with open(run_dir / "report_grouped.md", "w") as f:
            f.write(report)

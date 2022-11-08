"""Use this to run the pipeline with the test data."""
import json
import os
from pathlib import Path

from sacred.utils import recursive_update

smk_dir = "test_data/out/smk"

# Not configurable
fs_observer_dir = Path(smk_dir) / "runs"
data_root = "test_data/data_root/"
group_by = "test_data/supplementary_table.ods"
preprocessing_log = Path(smk_dir) / "preprocessing_log.csv"

# Configurable
classes = config["classes"]
experiment_name = config["experiment_name"]
config_updates = config["config_updates"]

# Class counts of all samples
file_stats_all_classes = Path(data_root) / "statistics_all_classes.csv"
# Class counts of subset according to selected classes
file_stats = Path(smk_dir) / "statistics_subset.csv"

# Assignment of samples to train/validation/test
file_best_indices = Path(data_root) / "partitioning.json"

smk_run_ids = config_updates.keys()


def get_config(wildcards):
    smk_run_id = wildcards.smk_run_id
    return f"{smk_dir}/smk_configs/{experiment_name}/{smk_run_id}.json"


rule all:
    input:
         expand(
             f"{smk_dir}/smk_flags/{config['experiment_name']}/evaluated/{{smk_run_id}}.smkflag",
             smk_run_id=smk_run_ids)


rule crop_tiles:
    """Crops patches from the tiles that can be input to the model."""
    input:
        group_by
    params:
          data_dir=Path(data_root) / "data"
    output:
          preprocessing_log,
          file_stats_all_classes
    shell:
         "python run_preprocessing.py --statsfile {file_stats_all_classes} --datapath {params.data_dir} --group-by-cases $'{input}' --padding"

rule subset_crops:
    """Create subset of samples based on selected classes that should be used for training."""
    input:
        file_stats_all_classes
    output:
          file_stats
    params:
        classes
    run:
         import os
         import pandas as pd

         # Symlink subset of classes to experiment data directory
         dir1 = os.path.dirname(file_stats_all_classes)
         dir2 = os.path.dirname(file_stats)
         os.makedirs(dir2)
         for _class in classes:
             assert os.path.exists(os.path.join(dir1, _class))
             os.symlink(os.path.join(dir1, _class), os.path.join(dir2, _class))

         # Update statistics without dropped classes
         df = pd.read_csv(file_stats_all_classes, index_col="Case")
         for col in df.columns:
             if col not in classes and col != "total":
                 df["total"] -= df[col]
                 df.drop(col, axis=1, inplace=True)
         df.to_csv(file_stats)

rule split_data:
    """Assign the samples to train, validation and test sets."""
    input:
         file_stats
    output:
          file_best_indices
    params:
          num_trials=5000
    shell:
        "python data_splitting.py {input} {output} {params.num_trials}"

rule configure:
    """Extracts the run configurations from the configfile."""
    output:
          f"{smk_dir}/smk_configs/{config['experiment_name']}/{{smk_run_id}}.json"
    run:
        with open(output[0],"w") as f:
            config_update = recursive_update(
                config["common_config"],config_updates[wildcards.smk_run_id])
            config_update["experiment_name"] = config["experiment_name"]
            json.dump(config_update,f,indent=4)

rule train:
    input:
        get_config,
        data_root,
        group_by
    output:
          touch(
              f"{smk_dir}/smk_flags/{config['experiment_name']}/trained/{{smk_run_id}}.smkflag"),
          # Mapping of snakemake config ID to sacred run ID
          f"{smk_dir}/smk_flags/{config['experiment_name']}/smk_to_sacred_id/{{smk_run_id}}"
    resources:
          gpus=1
    params:
        num_classes=config.get("num_classes",len(classes)),
        classes_to_keep=config["classes"]
    script:
          "run_training.py"


# Map snakemake's run ID to sacred's run ID
def smk_to_sacred_id(smk_run_id):
    with open(
            f"{smk_dir}/smk_flags/{config['experiment_name']}/smk_to_sacred_id/{{smk_run_id}}",
            "r") as f:
        sacred_run_id = f.read()
    return sacred_run_id


rule evaluate:
    input:
        f"{smk_dir}/smk_flags/{config['experiment_name']}/trained/{{smk_run_id}}.smkflag",
        f"{smk_dir}/smk_flags/{config['experiment_name']}/smk_to_sacred_id/{{smk_run_id}}"
    output:
        touch(
            f"{smk_dir}/smk_flags/{config['experiment_name']}/evaluated/{{smk_run_id}}.smkflag")
    script:
        "run_evaluation.py"

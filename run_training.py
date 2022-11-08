import json
import sys

import local_config
from main import ex

if __name__ == '__main__':
    # Read input arguments from snakemake
    try:
        config = snakemake.input[0]
        data_root = snakemake.input[1]
        group_samples = snakemake.input[2]
        num_classes = snakemake.params[0]
        classes_to_keep = snakemake.params[1]
    except NameError:
        # For debugging, script can be called without snakemake
        config = sys.argv[1]
        data_root = sys.argv[2]
        group_samples = sys.argv[3]
        num_classes = int(sys.argv[4])
        classes_to_keep = int(sys.argv[5]) if len(sys.argv) > 5 else None

    with open(config, "r") as f:
        config_updates = json.load(f)
    config_updates["data_root"] = data_root
    config_updates["group_samples"] = group_samples
    config_updates["num_classes"] = num_classes
    config_updates["classes_to_keep"] = classes_to_keep

    named_configs = list()
    options = {"--name": config_updates.pop("experiment_name")}
    local_config.add_observers(ex)
    run = ex.run(config_updates=config_updates,
                 named_configs=named_configs, options=options)

    with open(snakemake.output[1], "w") as f:
        f.write("None" if run._id is None else str(run._id))

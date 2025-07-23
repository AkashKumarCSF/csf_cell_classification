import local_config
import training
from experiment import ex

local_config.setup_logger(ex)


# noinspection PyUnusedLocal
@ex.config
def config():
    tags = []  # Omniboard tags
    num_epochs = 50
    batch_size = 32
    lr = 5e-5
    scheduler_type = "step"
    scheduler_kwargs = {
        "step": {
            "step_size": 10000,
            "gamma": 0.1
        },
        "cosine": {
            "T_max": 15
        }
    }[scheduler_type]
    optimizer_type = "adam"
    weight_decay = 0.0
    momentum = 0.9
    shuffle = True
    xval_interval = 1
    num_classes = 15
    reweight_classes = False
    early_stopping_metrics = ["auc"]
    patience = 5

    model_name = "vit_base_patch16_224"   # "resnet18"
    pretrained = True
    freeze_blocks = None
    cp_weight = 0

    # Data
    data_root = None
    group_samples = None
    img_size = 224
    crop_size = 224
    rescale = 1.0  # Factor by which image is rescaled before cropping
    blur_probability = .1
    max_blur_radius = 3.
    data_statistics = None
    split_file = "test_data/partitioning.yml"  # Train, val and test partitions
    subsample = 20000  # Maximum number of samples per class
    # Mapping of target class => original class
    class_groups = None  # Cannot use empty dict here because sacred will append
    group_during_training = False  # If true, classes are grouped during training, ie the model only learns the class groups. Otherwise, all classes are learned but the groups are used for evaluation.
    classes_to_keep = None  # If not None, continue only with these classes

    num_workers = 8
    log_dir_root = local_config.log_dir_root

    # Data augmentation
    rotation = 0
    color_jitter = {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.5,
    }

    break_after_first_inner_split = True
    break_after_first_outer_split = True
    # Load model from snapshot
    snapshot = False

    seed = 2938432

    group_by_case_in_val = True  # Group by patients in inner xval loop

    n_outer_splits = 1
    n_inner_splits = 3
    data_statistics = {
        "mean": [0.8178, 0.7881, 0.8823],
        "std": [0.2230, 0.2575, 0.1376]
    }


@ex.automain
def run_train_nn():
    return training.run_train_nn()

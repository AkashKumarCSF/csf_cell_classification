import efficientnet_pytorch
import numpy as np
import PIL
import sacred
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit

from dataloading import GroupedImageFolder, GroupedImageFolderWithClassGroups
from models import ConsistencyPriorModel

ex = sacred.Experiment("debug")


@ex.capture
def setup_model(model_name, pretrained, freeze_blocks, num_classes, cp_weight):
    if model_name == "efficientnetb4":
        # Not in torchvision zoo
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                               'nvidia_efficientnet_b4', pretrained=pretrained)
    elif model_name == "efficientnetb0":
        model = efficientnet_pytorch.EfficientNet.from_pretrained(
            "efficientnet-b0")
    else:
        model = {
            "resnet18": torchvision.models.resnet18,
            "resnet50": torchvision.models.resnet50,
            "vgg16": torchvision.models.vgg16,
            "densenet121": torchvision.models.densenet121,
        }[model_name](pretrained=pretrained)

    if freeze_blocks is not None:
        # ResNet18
        # 0: Conv
        # 4: Block1
        # 5: Block2
        # 6: Block3
        # 7: Block4
        # 9: Classification layer

        for m in list(model.children())[
                 :freeze_blocks]:
            for param in m.parameters():
                param.requires_grad = False

    # Reset last layer
    if model_name == "vgg16":
        num_features = list(model.classifier.children())[-1].in_features
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Linear(num_features, num_classes))
    elif model_name == "efficientnetb0":
        num_features = model._fc.in_features
        model._fc = nn.Linear(num_features, num_classes)
    elif model_name == "efficientnetb4":
        num_features = list(model.classifier.children())[-1].in_features
        model.classifier.fc = nn.Linear(num_features, num_classes)
    elif model_name == "densenet121":
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    else:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    if cp_weight is not None and cp_weight > 0:
        model = ConsistencyPriorModel(model)

    return model


@ex.capture
def setup_optimizer(model, lr, optimizer_type, weight_decay, momentum, scheduler_type, scheduler_kwargs):
    trainable_parameters = [param for name, param in model.named_parameters() if param.requires_grad]
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(trainable_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(trainable_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer {} not known!".format(optimizer_type))

    scheduler = {
        "step": torch.optim.lr_scheduler.StepLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR
    }[scheduler_type](optimizer, **scheduler_kwargs)

    return optimizer, scheduler


@ex.capture
def setup_transforms(crop_size, data_statistics, blur_probability,
                     max_blur_radius, color_jitter):
    class GaussianBlur:
        def __init__(self, probability, max_radius):
            self.probability = probability
            self.max_radius = max_radius

        def __call__(self, img):
            if np.random.uniform() < self.probability:
                radius = np.random.uniform(0, self.max_radius)
                img = img.filter(PIL.ImageFilter.GaussianBlur(radius))
            return img

    data_transforms = dict()
    data_transforms["train"] = [
        GaussianBlur(blur_probability, max_blur_radius),
        # Crop sample from WSI around annotation with translation margin
        transforms.CenterCrop(crop_size + 16),
        # Random translation
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(**color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(data_statistics["mean"], data_statistics["std"])
    ]
    data_transforms["xval"] = [
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(data_statistics["mean"], data_statistics["std"])
    ]
    data_transforms["test"] = data_transforms["xval"]
    data_transforms = {key: transforms.Compose(value) for key, value in data_transforms.items()}
    return data_transforms


@ex.capture
def get_dataset(data_root, group_samples, subsample, class_groups,
                group_during_training, classes_to_keep) -> GroupedImageFolder:
    if group_during_training and class_groups is not None:
        dataset = GroupedImageFolderWithClassGroups(class_groups, group_samples,
                                                    data_root,
                                                    classes_to_keep=classes_to_keep,
                                                    max_num=subsample)
    else:
        dataset = GroupedImageFolder(group_samples, data_root,
                                     classes_to_keep=classes_to_keep,
                                     max_num=subsample)

    # No duplicate classes
    assert len(dataset.classes) == len(set(dataset.classes))

    # logging.info(class_distribution(dataset, dataset.classes).to_markdown())

    return dataset


def kfold_split(dataset, n_splits, sub_idx=None, shuffle=True, random_state=0, can_split_train=1):
    if sub_idx is None:
        sub_idx = list(range(len(dataset.samples)))
    dummy_vals = np.zeros(len(sub_idx))
    targets = np.array([dataset.targets[i] for i in sub_idx])
    random_state = np.random.RandomState(random_state)

    # Make randomized kfold splits until they fulfill the requirements
    for i in range(100):
        if hasattr(dataset, 'groups'):
            groups = np.array([dataset.groups[i] for i in sub_idx])
            kf = GroupShuffleSplit(n_splits, test_size=1/n_splits, random_state=random_state.randint(1000))
            splits = list(kf.split(dummy_vals, targets, groups))
        else:
            kf = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state.randint(1000))
            splits = list(kf.split(dummy_vals, targets))

        try:
            for idx_train, idx_test in splits:
                for target in np.unique(targets):
                    n_train_groups_with_target = len({dataset.groups[sub_idx[idx]] for idx in idx_train
                                                      if dataset.targets[sub_idx[idx]] == target})
                    n_test_groups_with_target = len({dataset.groups[sub_idx[idx]] for idx in idx_test
                                                     if dataset.targets[sub_idx[idx]] == target})
                    # Relevant for nested crossvalidation:
                    # Every class should be can_split_train times in train set,
                    # so train set can be split again in can_split_train folds
                    # where each fold contains all classes
                    assert n_train_groups_with_target >= can_split_train
                    # Every class should be in test set
                    assert n_test_groups_with_target >= 1
            break
        except AssertionError:
            # Splits do not fulfill requirements, try again
            pass
    else:
        raise Exception("Could not make splits.")

    return splits

import logging
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from utils import map_orig_to_grouped, convert_classes


class SubsetWithTransform(torch.utils.data.Subset):
    def __init__(self, *args, data_transforms=None, **kwargs):
        self.data_transforms = data_transforms
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        sample, label, path = super().__getitem__(item)
        if self.data_transforms:
            sample = self.data_transforms(sample)
        return sample, label, path


class GroupedImageFolder(ImageFolder):
    """
    Every sample belongs to a group which should be considered for crossvalidation.

    Args:
        group_samples:
        max_num: If not None, each class will have a maximum of `max_num` samples (random downsampling).
    """

    def __init__(self, group_samples, *args, max_num: Optional[int] = None,
                 classes_to_keep: Optional[List[int]],
                 **kwargs):
        super().__init__(*args, **kwargs)
        df = pd.read_excel(group_samples,
                           engine='odf')  # engine odf for openoffice
        df = df.dropna(
            how='all')  # odf engine adds rows of all NaNs to end of table
        df = df.fillna(method='ffill')  # fill merged cells in table
        df["Case"] = df["Case"].astype(int)

        df = df.drop_duplicates(subset=["Index"])

        liquor_to_case = {liquor: df[df['Index'] == liquor]['Case'].item() for
                          liquor in df['Index']}
        try:
            self.groups = [
                liquor_to_case['_'.join(os.path.basename(path).split('_')[:2])]
                for path, _ in self.samples]
        except KeyError as e:
            key = e.args[0]
            if key.startswith('Liquor_'):
                try:
                    int(e.args[0])
                    raise ValueError(f'No case for {key} in {group_samples}.')
                except:
                    pass
            raise ValueError(f'Input images must have prefix Liquor_<number>_')

        def encode_dir_names(dir_name):
            return dir_name.encode("utf-8", "surrogateescape").decode("utf-8")

        self.classes = convert_classes(
            [encode_dir_names(c) for c in self.classes])

        if classes_to_keep is not None:
            self.remove_classes(classes_to_keep)

        self.max_num = max_num

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class and path is the relative path of the image.
        """
        sample, _ = super().__getitem__(index)
        target = self.targets[index]
        path, _ = self.samples[index]
        return sample, target, path

    def subsample(self, indices: np.array, max_num: int):
        """
        Limit the number of samples of each class by random subsampling if limit is exceeded, i.e. majority downsampling.

        indices: Indices of subset that are downsampled
        max_num: Maximum number of samples allowed per class.
        """
        imgs_subsampled = list()
        groups_subsampled = list()
        targets_subsampled = list()
        indices_new = list()

        for class_index in self.class_to_idx.values():
            # Get all samples of a class
            indices_class = [i for i, (sample, target) in enumerate(self.imgs)
                             if (target == class_index and i in indices)]
            imgs_class = [self.imgs[i] for i in indices_class]
            groups_class = [self.groups[i] for i in indices_class]
            targets_class = [self.targets[i] for i in indices_class]

            # Subsample if there are too many samples of this class
            if len(imgs_class) > max_num:
                indices_curr = np.random.choice(indices_class,
                                                size=max_num, replace=False)
                indices_new.extend(indices_curr.tolist())
            else:
                indices_new.extend(indices_class)

            imgs_subsampled.extend(imgs_class)
            groups_subsampled.extend(groups_class)
            targets_subsampled.extend(targets_class)

        logging.info(
            f"Subsampled from {len(indices)} to {len(indices_new)} samples (max. {max_num} samples per class).")

        if len(indices_new) == 0:
            raise ValueError(f"Removed all samples.")

        return indices_new

    def remove_classes(self, classes):
        """
        Exclude classes from the dataset even though they are present in the datafolder.

        classes: Classes to keep.
        """
        imgs_subsampled = list()
        groups_subsampled = list()
        targets_subsampled = list()
        class_to_idx_subsampled = dict()
        ctr = 0
        for class_, idx in self.class_to_idx.items():
            if class_ in classes:
                class_to_idx_subsampled[class_] = ctr
                ctr += 1
        assert list(class_to_idx_subsampled.values()) == list(
            range(len(classes))), class_to_idx_subsampled

        def get_key_by_val(val, d: dict):
            for key, val_curr in d.items():
                if val_curr == val:
                    break
            return key

        def update_target(target):
            """Convert old to new class index."""
            class_ = get_key_by_val(target, self.class_to_idx)
            new_idx = class_to_idx_subsampled[class_]
            return new_idx

        for i in range(len(self.imgs)):
            target = self.targets[i]

            if get_key_by_val(target, self.class_to_idx) in classes:
                target_new = update_target(target)
                targets_subsampled.append(target_new)
                imgs_subsampled.append((self.imgs[i][0], target_new))
                groups_subsampled.append(self.groups[i])

        logging.info(
            f"Subsampled from {len(self.imgs)} to {len(imgs_subsampled)} samples ({len(class_to_idx_subsampled)} of {len(self.class_to_idx)} classes).")

        # Make sure that we mapped to {0, ..., k}
        assert set(targets_subsampled) == set(range(len(classes))), set(
            targets_subsampled)
        self.imgs = imgs_subsampled
        self.groups = groups_subsampled
        self.samples = self.imgs
        self.targets = targets_subsampled
        self.class_to_idx = class_to_idx_subsampled
        self.classes = [get_key_by_val(t, self.class_to_idx) for t in
                        range(len(classes))]


def class_distribution(dataset: GroupedImageFolder, classes) -> pd.DataFrame:
    """Create dataframe of number of samples per class"""
    # targets = [target for sample, target, path in dataset]
    try:
        targets = dataset.targets
    except AttributeError:
        # Subset
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    targets = list(targets)
    df = dict()
    for i, c in enumerate(classes):
        df[i] = {
            "name": c,
            "freq": targets.count(i),
        }
    df = pd.DataFrame(df).T
    df["ratio"] = df["freq"] / df["freq"].sum()
    return df


class GroupedImageFolderWithClassGroups(GroupedImageFolder):
    def __init__(self, class_groups, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class_int_mapping, orig_to_grouped = map_orig_to_grouped(self.classes,
                                                                 class_groups)

        def map_names(target: int):
            orig_class_name = self.classes[target]
            grouped_class_name = orig_to_grouped[orig_class_name]
            grouped_target = class_int_mapping[grouped_class_name]
            return grouped_target

        self.targets = [map_names(target) for target in self.targets]
        self.classes = list(class_int_mapping.keys())
        self.class_groups = class_groups


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class and path is the relative path of the image.
        """
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path


def compute_mean_std(data_root):
    dataset = ImageFolder(data_root)
    samples = list()
    for i, (data, target, path) in tqdm(enumerate(dataset), total=len(dataset)):
        data = to_tensor(data)
        assert data.shape[1] == 224 and data.shape[2] == 224, dataset.samples[i]
        samples.append(data)
    samples = torch.stack(samples, dim=0)
    mean = torch.mean(samples, dim=(0, 2, 3))
    std = torch.std(samples, dim=(0, 2, 3))
    print("Mean:", mean)
    print("Std:", std)


def compute_class_weights(data_root, num_classes):
    dataset = ImageFolder(data_root)
    inverse_relative_counts = {t: len(dataset.targets)/dataset.targets.count(t) for t in range(num_classes)}
    normalization = sum(inverse_relative_counts.values())
    weight_sum = 0
    for target in range(num_classes):
        print(f"Class weight of {target}:", round(inverse_relative_counts[target] / normalization, 4))
        weight_sum += round(inverse_relative_counts[target] / normalization, 4)
    print(f"Sum of weights: {weight_sum}")

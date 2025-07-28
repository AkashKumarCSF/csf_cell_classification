import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import json
from main import config
import argparse
from pathlib import Path
from collections import Counter

from surgeon_pytorch import Inspect, get_layers

from experiment import setup_model, get_dataset, setup_transforms
from dataloading import SubsetWithTransform

"""
parser = argparse.ArgumentParser()
parser.add_argument('--smk_run_id', type=str, required=True)
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--smk_dir', type=str, required=True)
args = parser.parse_args()
"""

# === CONFIG ===
config_path = "test_data/out/smk_configs/csf_classes_v02_Ex3/0.json"
snapshot_path = "test_data/out/runs/Ex3/model-bal_acc.pt"
split_file = "test_data/split_file.json"
data_root = "/home/adminuser/backup/work_directory/zenodo/"
group_samples = "test_data/supplementary_table_73.ods"

model_name = "vit_base_patch16_224"
pretrained = False
freeze_blocks = None
num_classes = 15
cp_weight = 0.8
batch_size = 32

# Load the JSON config
with open(config_path, 'r') as f:
    sacred_config = json.load(f)

crop_size = 224
data_statistics = {
        "mean": [0.8178, 0.7881, 0.8823],
        "std": [0.2230, 0.2575, 0.1376]
    }
blur_probability = 0.1
max_blur_radius = 3.
color_jitter = {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.5,
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
model = setup_model(model_name, pretrained, freeze_blocks, num_classes, cp_weight)

checkpoint = torch.load(snapshot_path, map_location=device)

# If saved with 'state_dict', use that
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

# Strip 'module.' prefix from keys
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

# Load cleaned state dict into model
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# ✅ For ConsistencyPriorModel: access internal model
inner_model = model.network if hasattr(model, 'network') else model

# === INSPECT LAYER NAMES ===
#for name in get_layer_names(inner_model):
#    print(" ", name)

# === SET FEATURE EXTRACTOR ===
if model_name == "vit_base_patch16_224":
    def extract_features(x):
        return inner_model.forward_features(x)
    print("[ViT] Using model.forward_features() to extract embeddings.")
else:
    feature_extractor = nn.Sequential(
        inner_model.network.features,
        inner_model.network.avgpool,
        nn.Flatten(1),
        *list(model.network.classifier.children())[:-1]
    )
    def extract_features(x):
        return feature_extractor(x)
    print("[CNN] Using intermediate CNN layers for embedding extraction.")



# === LOAD DATASET ===

data_transforms = setup_transforms(
    crop_size=crop_size,
    data_statistics=data_statistics,
    blur_probability=blur_probability,
    max_blur_radius=max_blur_radius,
    color_jitter=color_jitter
)


dataset = get_dataset(data_root=data_root,
                      group_samples=group_samples,
                      subsample=20000,
                      class_groups=None,
                      group_during_training=False,
                      classes_to_keep=None)
classes = dataset.classes

with open(split_file) as f:
    pid_splits = json.load(f)
_, _, idx_test = [
    [i for i, pid in enumerate(dataset.groups) if str(pid) in map(str, pid_splits[split])]
    for split in ["train", "val", "test"]
]

test_set = SubsetWithTransform(dataset, idx_test,
                               data_transforms=data_transforms["test"])
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# === EXTRACT ACTIVATIONS ===
all_activations = []
all_labels = []

print("Extracting layer activations...")
with torch.no_grad():
    for data, target, _ in test_loader:
        data = data.to(device)
        target = target.to(device)

        activations = extract_features(data)

        if activations.dim() > 2:
            activations = torch.flatten(activations, start_dim=1)  # Flatten spatial dims

        all_activations.append(activations.cpu().numpy())
        all_labels.append(target.cpu().numpy())

all_activations = np.concatenate(all_activations, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

label_counts = Counter(all_labels)
print("\n✅ Class distribution in test set (actual counts):")
for i, class_name in enumerate(classes):
    count = label_counts.get(i, 0)
    print(f"{class_name}: {count} actual")

# === PCA ===
print("Running PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_activations)

# === PLOT ===
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10.colors if len(classes) <= 10 else plt.cm.rainbow(np.linspace(0, 1, len(classes)))
for i, class_name in enumerate(classes):
    idxs = all_labels == i
    marker = '^' if class_name.lower() == "tumo" else 'o'
    plt.scatter(
        pca_result[idxs, 0], pca_result[idxs, 1],
        label=class_name,
        alpha=0.6,
        color=colors[i % len(colors)],
        marker=marker
    )
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of ViT embeddings (CLS token)" if model_name.startswith("vit") else "PCA of CNN pre-logits")
plt.legend()
plt.tight_layout()
plt.show()

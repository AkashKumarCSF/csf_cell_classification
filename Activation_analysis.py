import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image
from tqdm import tqdm

from experiment import setup_model, setup_transforms
from main import config  # If needed for shared constants like mean/std

# === CONFIGURATION ===
snapshot_path = "test_data/out/runs/Ex3/model-bal_acc.pt"
results_csv = "test_data/out/runs/Ex3/predictions_test.csv"
model_name = "vit_base_patch16_224"
pretrained = False
freeze_blocks = None
num_classes = 15
cp_weight = 0.8
crop_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_statistics = {
    "mean": [0.8178, 0.7881, 0.8823],
    "std": [0.2230, 0.2575, 0.1376]
}
color_jitter = {k: 0 for k in ["brightness", "contrast", "saturation", "hue"]}

# === LOAD MODEL ===
model = setup_model(model_name, pretrained, freeze_blocks, num_classes, cp_weight)
checkpoint = torch.load(snapshot_path, map_location=device)
checkpoint = checkpoint.get('state_dict', checkpoint)
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()
inner_model = model.network if hasattr(model, 'network') else model

# === EXTRACT FEATURE FUNCTION ===
if model_name.startswith("vit"):
    def extract_features(x):
        return inner_model.forward_features(x)
else:
    raise NotImplementedError("Only ViT is supported in this cleaned script.")

# === LOAD TEST CSV AND MAP CLASSES ===
df = pd.read_csv(results_csv)
class_names = sorted(df['filename'].apply(lambda x: Path(x).parts[-2]).unique())
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
print("✅ Classes:", class_to_idx)

# === DEFINE TEST TRANSFORM ===
test_transform = setup_transforms(
    crop_size=crop_size,
    data_statistics=data_statistics,
    blur_probability=0.0,
    max_blur_radius=0.0,
    color_jitter=color_jitter
)["test"]

# === EXTRACT ACTIVATIONS FROM IMAGES ===
all_activations = []
all_labels = []

print("Extracting activations...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row['filename']
    label_name = Path(img_path).parts[-2]
    label = class_to_idx[label_name]

    img = Image.open(img_path).convert("RGB")
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = extract_features(img_tensor)
        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)
    all_activations.append(features.cpu().numpy())
    all_labels.append(label)

all_activations = np.concatenate(all_activations, axis=0)
all_labels = np.array(all_labels)

# === CLASS COUNTS ===
label_counts = Counter(all_labels)
print("\n✅ Class distribution:")
for class_name, idx in class_to_idx.items():
    print(f"{class_name}: {label_counts[idx]} samples")

# === PCA ===
print("Running PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_activations)

# === PLOT PCA WITH TUMOR IN RED AND TRIANGLE ===
plt.figure(figsize=(8, 6))

# Assign color for each class: red for Tumor, colormap for others
fixed_colors = {}
tumor_color = 'red'
colormap = plt.cm.tab10.colors if len(class_names) <= 10 else plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
color_idx = 0

for class_name in class_names:
    if class_name.lower() == "tumor":
        fixed_colors[class_name] = tumor_color
    else:
        fixed_colors[class_name] = colormap[color_idx % len(colormap)]
        color_idx += 1

# Plot each class
for class_name, idx in class_to_idx.items():
    marker = '^' if class_name.lower() == "tumor" else 'o'
    mask = all_labels == idx
    plt.scatter(
        pca_result[mask, 0], pca_result[mask, 1],
        label=class_name,
        color=fixed_colors[class_name],
        alpha=0.6,
        marker=marker
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of ViT Embeddings (CLS Token)")
plt.legend()
plt.tight_layout()
plt.show()


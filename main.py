import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from typing import List
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet18 (feature extractor)
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC
model.eval().to(device)

# Preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load frames from folder ===
def load_frames(folder: str):
    images, filenames = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".jpg"):
            img_path = os.path.join(folder, fname)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            images.append(img)
            filenames.append(fname)
    return images, filenames

# === Feature Extraction ===
def extract_features(images: List[np.ndarray]):
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="Extracting features"):
            inp = preprocess(img).unsqueeze(0).to(device)
            feat = model(inp).squeeze().cpu().numpy()
            features.append(feat)
    return np.array(features)

# === Compute pairwise cosine distances ===
def compute_distances(features: np.ndarray):
    return cdist(features, features, metric="cosine")

# # === Greedy TSP-based global sorting ===
# def global_sort(dist_matrix: np.ndarray):
    n = dist_matrix.shape[0]
    visited = [0]
    for _ in range(n - 1):
        last = visited[-1]
        dists = dist_matrix[last]
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        visited.append(next_node)
    return visited

import random

# === Random Seed Sort: builds sequence linearly from a random start
def random_seed_sort(dist_matrix: np.ndarray):
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    seed = random.choice(list(unvisited))
    
    order = [seed]
    unvisited.remove(seed)

    while unvisited:
        last = order[-1]
        dists = dist_matrix[last]
        
        # Mask already visited
        dists = [(i, dists[i]) for i in unvisited]
        next_node = min(dists, key=lambda x: x[1])[0]

        order.append(next_node)
        unvisited.remove(next_node)
    
    return order


def best_of_n_random_seeds(dist_matrix, n_trials=10):
    best_order = None
    best_cost = float('inf')
    
    for _ in range(n_trials):
        order = random_seed_sort(dist_matrix)
        cost = sum(dist_matrix[order[i], order[i+1]] for i in range(len(order)-1))
        if cost < best_cost:
            best_cost = cost
            best_order = order
    return best_order



# === Save reordered frames ===
def save_ordered(images, order, filenames, output_folder="ordered_frames"):
    os.makedirs(output_folder, exist_ok=True)
    for idx, i in enumerate(order):
        out_path = os.path.join(output_folder, f"{idx:03d}_{filenames[i]}")
        cv2.imwrite(out_path, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
    print(f"✅ Saved ordered frames to '{output_folder}'")

# === Updated Accuracy Check (for '1.jpg', '2.jpg'... filenames) ===
def compute_accuracy(predicted_folder: str, ground_truth_folder: str):
    pred_files = sorted([f for f in os.listdir(predicted_folder) if f.endswith(".jpg")])
    gt_files = sorted(os.listdir(ground_truth_folder), key=lambda x: int(x.split('.')[0]))

    if len(pred_files) != len(gt_files):
        print(f"⚠️ Mismatch in frame counts: predicted = {len(pred_files)}, ground truth = {len(gt_files)}")
        return

    scores = []
    for p, g in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Comparing frames"):
        pred_path = os.path.join(predicted_folder, p)
        gt_path = os.path.join(ground_truth_folder, g)

        pred_img = cv2.imread(pred_path)
        gt_img = cv2.imread(gt_path)

        if pred_img is None or gt_img is None:
            continue

        pred_gray = cv2.resize(cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY), (224, 224))
        gt_gray = cv2.resize(cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY), (224, 224))

        sim = ssim(pred_gray, gt_gray)
        scores.append(sim)

    acc = np.mean(scores) * 100
    print(f"✅ SSIM-based Accuracy (1.jpg, 2.jpg naming): {acc:.2f}%")


# === Main Pipeline ===
if __name__ == "__main__":
    shuffled_folder = "shuffled_frames"
    ground_truth_folder = "correct_frames"
    output_folder = "ordered_frames"

    # Step 1: Load and extract features
    images, filenames = load_frames(shuffled_folder)
    features = extract_features(images)

    # Step 2: Pairwise distances + sorting
    dist_matrix = compute_distances(features)
    order = best_of_n_random_seeds(dist_matrix)


    # Step 3: Save ordered frames
    save_ordered(images, order, filenames, output_folder)

    # Step 4: Accuracy evaluation
    compute_accuracy(output_folder, ground_truth_folder)



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

# Loading pre-trained DenseNet121 for feature extraction
from torchvision.models import densenet121, DenseNet121_Weights
backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features
gap = torch.nn.AdaptiveAvgPool2d((1, 1))  # Global Avg Pool
model = torch.nn.Sequential(backbone, gap)
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
def compute_hybrid_distances(features, images, top_k=30):
    cosine_dist = cdist(features, features, metric="cosine")
    n = len(images)
    
    # Pre-convert to grayscale
    gray_images = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (112, 112)) 
                   for img in images]
    
    # Compute SSIM for ALL pairs, but prioritize closest ones
    ssim_dist = np.ones((n, n))  # Initialize with max distance
    
    # Compute SSIM for top-k pairs per frame
    computed_pairs = set()
    for i in tqdm(range(n), desc="Computing SSIM"):
        closest_indices = np.argsort(cosine_dist[i])[:top_k+1]
        for j in closest_indices:
            if i != j and (i,j) not in computed_pairs and (j,i) not in computed_pairs:
                ssim_val = ssim(gray_images[i], gray_images[j])
                ssim_dist[i,j] = ssim_dist[j,i] = 1 - ssim_val
                computed_pairs.add((i,j))
    
    # For uncomputed pairs, use cosine distance
    for i in range(n):
        for j in range(n):
            if ssim_dist[i,j] == 1.0 and i != j:  # Not computed
                ssim_dist[i,j] = cosine_dist[i,j]
    
    # Weighted combination
    alpha = 0.7
    return alpha * cosine_dist + (1-alpha) * ssim_dist



#legacy method
# # === Greedy TSP-based global sorting ===
# def global_sort(dist_matrix: np.ndarray):
#     n = dist_matrix.shape[0]
#     visited = [0]
#     for _ in range(n - 1):
#         last = visited[-1]
#         dists = dist_matrix[last]
#         dists[visited] = np.inf
#         next_node = np.argmin(dists)
#         visited.append(next_node)
#     return visited



# === Random Seed Sort: builds sequence linearly from a random start
def random_seed_sort(dist_matrix: np.ndarray, start_idx: int):
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    order = [start_idx]
    unvisited.remove(start_idx)
    while unvisited:
        last = order[-1]
        dists = dist_matrix[last]
        dists = [(i, dists[i]) for i in unvisited]
        next_node = min(dists, key=lambda x: x[1])[0]
        order.append(next_node)
        unvisited.remove(next_node)
    return order



# === Save reordered frames ===
def save_ordered(images, order, filenames, output_folder="ordered_frames"):
    os.makedirs(output_folder, exist_ok=True)
    for idx, i in enumerate(order):
        out_path = os.path.join(output_folder, f"{idx:03d}_{filenames[i]}")
        cv2.imwrite(out_path, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
    print(f" Saved ordered frames to '{output_folder}'")



# === Main Pipeline ===
if __name__ == "__main__":
    shuffled_folder = "shuffled_frames"
    output_folder = "ordered_frames"

    # Step 1: Load and extract features
    images, filenames = load_frames(shuffled_folder)
    features = extract_features(images)

    # Step 2: Pairwise distances + sorting
    dist_matrix = compute_hybrid_distances(features,images)
    start_filename = "4096000.jpg"
    start_idx = filenames.index(start_filename)
    order = random_seed_sort(dist_matrix, start_idx)



    save_ordered(images, order, filenames, output_folder)


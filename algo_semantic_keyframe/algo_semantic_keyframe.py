"""
Stratégie : Semantic keyframe selection (CLIP + KMeans)
Compatible orchestrateur (CLI + contraintes globales)

Entrées :
    --video   : chemin vidéo
    --frames  : nombre de frames à extraire
    --output  : dossier de sortie
    --width   : (optionnel) resize largeur
    --height  : (optionnel) resize hauteur
"""

import argparse
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel

# =====================
# ARGPARSE (piloté par orchestrateur)
# =====================

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--frames", type=int, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--budget", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--height", type=int, default=None)
args = parser.parse_args()

VIDEO_PATH    = args.video
TARGET_FRAMES = args.frames
OUTPUT_DIR    = args.output
TARGET_WIDTH  = args.width
TARGET_HEIGHT = args.height

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# CONFIG
# =====================

NUM_SAMPLED = 128
BATCH_SIZE  = 16
CLIP_MODEL  = "openai/clip-vit-large-patch14"
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "viz_semantic.png")

# =====================
# VIDEO INFO + SAMPLING
# =====================

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps          = cap.get(cv2.CAP_PROP_FPS)
orig_width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_s         = total_frames_video / video_fps

print(f"Video: {VIDEO_PATH}")
print(f"Target frames: {TARGET_FRAMES}")
print(f"Sampling: {NUM_SAMPLED}")

num_sampled = min(NUM_SAMPLED, total_frames_video)
indices = np.linspace(0, total_frames_video - 1, num_sampled, dtype=int)

frames_pil = []
frames_cv2 = []
frame_ids  = []

for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(rgb))
        frames_cv2.append(frame)
        frame_ids.append(idx)

cap.release()

# =====================
# CLIP ENCODING
# =====================

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()

def encode_frames(frames):
    embs = []
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i:i+BATCH_SIZE]
        inputs = clip_processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            # Some transformers builds can return a model output object instead of a tensor.
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0, :]
            feats = torch.nn.functional.normalize(feats, dim=-1)
        embs.append(feats.cpu().numpy())
    return np.concatenate(embs, axis=0)

embeddings = encode_frames(frames_pil)

# =====================
# KMEANS
# =====================

k = min(TARGET_FRAMES, len(frames_pil))

kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = kmeans.fit_predict(embeddings)
centers = kmeans.cluster_centers_

# =====================
# SELECTION
# =====================

selected = []

for i in range(k):
    cluster_idx = np.where(labels == i)[0]

    if len(cluster_idx) == 0:
        continue

    emb_cluster = embeddings[cluster_idx]
    d = np.linalg.norm(emb_cluster - centers[i], axis=1)
    best = cluster_idx[np.argmin(d)]

    selected.append((frame_ids[best], frames_cv2[best]))

selected.sort(key=lambda x: x[0])

# =====================
# SAVE (avec resize)
# =====================

extracted = []

for i, (frame_id, frame) in enumerate(selected):

    if TARGET_WIDTH and TARGET_HEIGHT:
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.jpg")
    cv2.imwrite(path, frame)

    extracted.append((frame_id, frame))

# =====================
# VISUALIZATION
# =====================

cols = 4
rows = math.ceil(len(extracted)/cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
fig.suptitle(f"Semantic CLIP KMeans — {len(extracted)} frames", fontsize=12)

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, (frame_id, frame) in enumerate(extracted):
    axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"{frame_id}")
    axes[i].axis("off")

for j in range(len(extracted), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PLOT)

# =====================
# TOKEN ESTIMATION
# =====================

def estimate_tokens(n, w, h, patch=28):
    w = round(w/patch)*patch
    h = round(h/patch)*patch
    return n * (w//patch) * (h//patch)

final_w = TARGET_WIDTH if TARGET_WIDTH else orig_width
final_h = TARGET_HEIGHT if TARGET_HEIGHT else orig_height

tokens = estimate_tokens(len(extracted), final_w, final_h)

print("="*50)
print("DONE")
print(f"Frames: {len(extracted)}")
print(f"Resolution: {final_w}x{final_h}")
print(f"Tokens: {tokens}")
print(f"Output: {OUTPUT_DIR}")
print("="*50)
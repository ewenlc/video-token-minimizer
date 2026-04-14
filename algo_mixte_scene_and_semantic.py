"""
Stratégie : Scene-aware + Semantic keyframe selection (mixte)
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
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

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

THRESHOLD         = 27.0
BRIGHTNESS_THRESH = 15
FRAMES_PER_SCENE_SAMPLE = 16
BATCH_SIZE        = 16
CLIP_MODEL        = "openai/clip-vit-large-patch14"
OUTPUT_PLOT       = os.path.join(OUTPUT_DIR, "viz_scene_semantic.png")

# =====================
# VIDEO INFO
# =====================

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps          = cap.get(cv2.CAP_PROP_FPS)
orig_width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {VIDEO_PATH}")
print(f"Target frames: {TARGET_FRAMES}")
print(f"Resolution: {orig_width}x{orig_height}")

# =====================
# SCENE DETECTION
# =====================

video_sd  = open_video(VIDEO_PATH)
scene_mgr = SceneManager()
scene_mgr.add_detector(ContentDetector(threshold=THRESHOLD))
scene_mgr.detect_scenes(video_sd)
scenes    = scene_mgr.get_scene_list()

if len(scenes) == 0:
    scenes_frames = [(0, total_frames_video - 1)]
else:
    scenes_frames = [(s[0].get_frames(), s[1].get_frames() - 1) for s in scenes]

# =====================
# FILTER DARK SCENES
# =====================

filtered = []
for start, end in scenes_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, (start + end) // 2)
    ret, frame = cap.read()
    brightness = np.mean(frame) if (ret and frame is not None) else 0
    if brightness >= BRIGHTNESS_THRESH:
        filtered.append((start, end))

if len(filtered) == 0:
    filtered = [(0, total_frames_video - 1)]

scenes_frames = filtered

# =====================
# DISTRIBUTE FRAMES
# =====================

nb_scenes = len(scenes_frames)

if nb_scenes >= TARGET_FRAMES:
    scenes_with_len = sorted(
        [(s[0], s[1], s[1]-s[0]) for s in scenes_frames],
        key=lambda x: x[2],
        reverse=True
    )[:TARGET_FRAMES]
    scenes_to_process = [(s[0], s[1], 1) for s in scenes_with_len]
    scenes_to_process.sort(key=lambda x: x[0])
else:
    frames_per_scene = [1] * nb_scenes
    remaining = TARGET_FRAMES - nb_scenes
    lengths = [s[1] - s[0] for s in scenes_frames]

    for _ in range(remaining):
        scores = [lengths[i] / (frames_per_scene[i] + 1) for i in range(nb_scenes)]
        best = scores.index(max(scores))
        frames_per_scene[best] += 1

    scenes_to_process = [
        (scenes_frames[i][0], scenes_frames[i][1], frames_per_scene[i])
        for i in range(nb_scenes)
    ]

# =====================
# CLIP
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

# =====================
# SELECTION
# =====================

selected = []

for start, end, n_select in scenes_to_process:
    indices = np.linspace(start, end, min(FRAMES_PER_SCENE_SAMPLE, end-start+1), dtype=int)

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

    if len(frames_pil) == 0:
        continue

    if len(frames_pil) <= n_select:
        selected.extend([(frame_ids[i], frames_cv2[i]) for i in range(len(frames_pil))])
        continue

    emb = encode_frames(frames_pil)

    if n_select == 1:
        centroid = emb.mean(axis=0)
        idx = np.argmin(np.linalg.norm(emb - centroid, axis=1))
        selected.append((frame_ids[idx], frames_cv2[idx]))
    else:
        kmeans = KMeans(n_clusters=n_select, n_init="auto").fit(emb)
        for i in range(n_select):
            cluster_idx = np.where(kmeans.labels_ == i)[0]
            center = kmeans.cluster_centers_[i]
            d = np.linalg.norm(emb[cluster_idx] - center, axis=1)
            best = cluster_idx[np.argmin(d)]
            selected.append((frame_ids[best], frames_cv2[best]))

# =====================
# SORT
# =====================

selected.sort(key=lambda x: x[0])

# =====================
# SAVE (avec contrainte spatiale)
# =====================

extracted = []

for i, (frame_id, frame) in enumerate(selected):

    if TARGET_WIDTH and TARGET_HEIGHT:
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.jpg")
    cv2.imwrite(path, frame)
    extracted.append(frame)

cap.release()

# =====================
# VISUALIZATION
# =====================

cols = 4
rows = math.ceil(len(extracted)/cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, frame in enumerate(extracted):
    axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
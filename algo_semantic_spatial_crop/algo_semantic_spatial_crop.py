"""
Stratégie : Semantic spatial crop (YOLO) + uniform selection
Compatible orchestrateur (CLI + contraintes globales)

Entrées :
    --video   : chemin vidéo
    --frames  : (optionnel) nombre de frames cible (fallback)
    --output  : dossier de sortie
    --budget  : budget tokens (prioritaire)
    --width   : (optionnel) resize final
    --height  : (optionnel) resize final
"""

import argparse
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# =====================
# ARGPARSE
# =====================

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--frames", type=int, default=None)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--budget", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--height", type=int, default=None)
args = parser.parse_args()

VIDEO_PATH   = args.video
TARGET_FRAMES = args.frames
TOKEN_BUDGET = args.budget
OUTPUT_DIR   = args.output
TARGET_WIDTH = args.width
TARGET_HEIGHT = args.height

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# CONFIG
# =====================

PATCH_SIZE   = 28
MARGIN       = 20
PROBE_FRAMES = 64
YOLO_CONF    = 0.25
OUTPUT_PLOT  = os.path.join(OUTPUT_DIR, "viz_yolo_crop.png")

# =====================
# VIDEO INFO
# =====================

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {VIDEO_PATH}")
print(f"Budget: {TOKEN_BUDGET}")
print(f"Resolution: {orig_w}x{orig_h}")

# =====================
# YOLO LOAD
# =====================

from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")

# =====================
# UTILS
# =====================

def crop_person(frame):
    results = yolo(frame, conf=YOLO_CONF, classes=[0], verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return np.zeros_like(frame), False

    coords = boxes.xyxy.cpu().numpy()

    x1 = int(max(0, coords[:,0].min() - MARGIN))
    y1 = int(max(0, coords[:,1].min() - MARGIN))
    x2 = int(min(orig_w, coords[:,2].max() + MARGIN))
    y2 = int(min(orig_h, coords[:,3].max() + MARGIN))

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros_like(frame), False

    return crop, True


def estimate_tokens_frame(frame):
    h, w = frame.shape[:2]
    w = round(w/PATCH_SIZE)*PATCH_SIZE
    h = round(h/PATCH_SIZE)*PATCH_SIZE
    return (w//PATCH_SIZE) * (h//PATCH_SIZE)

# =====================
# PROBE PASS (estimation tokens/frame)
# =====================

probe_indices = np.linspace(0, total_frames-1, PROBE_FRAMES, dtype=int)

tokens_list = []

for idx in probe_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue

    crop, _ = crop_person(frame)
    tokens = estimate_tokens_frame(crop)
    tokens_list.append(tokens)

avg_tokens = int(np.mean(tokens_list)) if tokens_list else 1

print(f"Avg tokens/frame: {avg_tokens}")

# =====================
# FRAME COUNT DECISION
# =====================

if TOKEN_BUDGET:
    nb_frames = max(1, TOKEN_BUDGET // avg_tokens)
elif TARGET_FRAMES:
    nb_frames = TARGET_FRAMES
else:
    nb_frames = 16

nb_frames = min(nb_frames, total_frames)

print(f"Selected frames count: {nb_frames}")

# =====================
# FINAL SAMPLING
# =====================

selected_ids = np.linspace(0, total_frames-1, nb_frames, dtype=int)

# =====================
# EXTRACTION
# =====================

extracted = []
tokens_real = []

for i, fid in enumerate(selected_ids):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if not ret:
        continue

    crop, detected = crop_person(frame)

    # 🔥 contrainte globale
    if TARGET_WIDTH and TARGET_HEIGHT:
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

    t = estimate_tokens_frame(crop)
    tokens_real.append(t)

    path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.jpg")
    cv2.imwrite(path, crop)

    extracted.append((fid, crop, detected, t))

cap.release()

# =====================
# VISUALIZATION
# =====================

cols = 4
rows = math.ceil(len(extracted)/cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
fig.suptitle(f"YOLO crop — {len(extracted)} frames", fontsize=12)

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, (fid, frame, detected, t) in enumerate(extracted):
    axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status = "✓" if detected else "✗"
    axes[i].set_title(f"{fid} | {t} tok {status}")
    axes[i].axis("off")

for j in range(len(extracted), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PLOT)

# =====================
# SUMMARY
# =====================

total_tokens = sum(tokens_real)

print("="*50)
print("DONE")
print(f"Frames: {len(extracted)}")
print(f"Total tokens: {total_tokens}")
print(f"Avg tokens/frame: {int(np.mean(tokens_real))}")
print(f"Budget usage: {total_tokens / TOKEN_BUDGET * 100:.1f}%" if TOKEN_BUDGET else "")
print(f"Output: {OUTPUT_DIR}")
print("="*50)
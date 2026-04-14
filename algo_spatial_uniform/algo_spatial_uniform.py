"""
Stratégie : Spatial uniform frame selection (resolution-driven)
Compatible orchestrateur (CLI + contraintes globales)

Entrées :
    --video   : chemin vidéo
    --frames  : (optionnel) fallback si pas de budget
    --output  : dossier de sortie
    --budget  : budget tokens (prioritaire)
    --width   : résolution cible largeur (obligatoire si budget)
    --height  : résolution cible hauteur (obligatoire si budget)
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

PATCH_SIZE  = 28
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "viz_spatial_uniform.png")

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
print(f"Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")

# =====================
# RESOLUTION + TOKENS
# =====================

def round_patch(x):
    return max(PATCH_SIZE, round(x / PATCH_SIZE) * PATCH_SIZE)

if TARGET_WIDTH and TARGET_HEIGHT:
    resized_w = round_patch(TARGET_WIDTH)
    resized_h = round_patch(TARGET_HEIGHT)
else:
    resized_w = round_patch(orig_w)
    resized_h = round_patch(orig_h)

tokens_per_frame = (resized_w // PATCH_SIZE) * (resized_h // PATCH_SIZE)

# =====================
# FRAME COUNT DECISION
# =====================

if TOKEN_BUDGET:
    nb_frames = max(1, TOKEN_BUDGET // tokens_per_frame)
elif TARGET_FRAMES:
    nb_frames = TARGET_FRAMES
else:
    nb_frames = 16

nb_frames = min(nb_frames, total_frames)

print(f"Tokens/frame: {tokens_per_frame}")
print(f"Selected frames: {nb_frames}")

# =====================
# SAMPLING
# =====================

selected_ids = np.linspace(0, total_frames-1, nb_frames, dtype=int)

# =====================
# EXTRACTION + RESIZE
# =====================

extracted = []

for i, fid in enumerate(selected_ids):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if not ret:
        continue

    # 🔥 contrainte globale (spatial driver)
    frame = cv2.resize(frame, (resized_w, resized_h))

    path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.jpg")
    cv2.imwrite(path, frame)

    extracted.append((fid, frame))

cap.release()

# =====================
# VISUALIZATION
# =====================

cols = 4
rows = math.ceil(len(extracted)/cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
fig.suptitle(f"Spatial uniform — {len(extracted)} frames", fontsize=12)

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, (fid, frame) in enumerate(extracted):
    axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"{fid}")
    axes[i].axis("off")

for j in range(len(extracted), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PLOT)

# =====================
# SUMMARY
# =====================

total_tokens = len(extracted) * tokens_per_frame

print("="*50)
print("DONE")
print(f"Frames: {len(extracted)}")
print(f"Resolution: {resized_w}x{resized_h}")
print(f"Tokens/frame: {tokens_per_frame}")
print(f"Total tokens: {total_tokens}")
print(f"Budget usage: {total_tokens / TOKEN_BUDGET * 100:.1f}%" if TOKEN_BUDGET else "")
print(f"Output: {OUTPUT_DIR}")
print("="*50)
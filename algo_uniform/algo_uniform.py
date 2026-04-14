"""
Stratégie : Uniform frame selection
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

# =====================
# ARGPARSE
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
# VIDEO INFO
# =====================

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {VIDEO_PATH}")
print(f"Target frames: {TARGET_FRAMES}")

# =====================
# SAMPLING
# =====================

selected_ids = np.linspace(0, total_frames-1, TARGET_FRAMES, dtype=int)

# =====================
# EXTRACTION (+ resize)
# =====================

extracted = []

for i, fid in enumerate(selected_ids):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()

    if not ret:
        continue

    # 🔥 contrainte globale
    if TARGET_WIDTH and TARGET_HEIGHT:
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.jpg")
    cv2.imwrite(path, frame)

    extracted.append((fid, frame))

cap.release()

# =====================
# VISUALIZATION
# =====================

cols = 4
rows = math.ceil(len(extracted)/cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
fig.suptitle(f"Uniform — {len(extracted)} frames", fontsize=12)

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, (fid, frame) in enumerate(extracted):
    axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"{fid}")
    axes[i].axis("off")

for j in range(len(extracted), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_uniform.png"))

# =====================
# TOKEN ESTIMATION
# =====================

def estimate_tokens(n, w, h, patch=28):
    w = round(w/patch)*patch
    h = round(h/patch)*patch
    return n * (w//patch) * (h//patch)

final_w = TARGET_WIDTH if TARGET_WIDTH else orig_w
final_h = TARGET_HEIGHT if TARGET_HEIGHT else orig_h

tokens = estimate_tokens(len(extracted), final_w, final_h)

# =====================
# SUMMARY
# =====================

print("="*50)
print("DONE")
print(f"Frames: {len(extracted)}")
print(f"Resolution: {final_w}x{final_h}")
print(f"Tokens: {tokens}")
print(f"Output: {OUTPUT_DIR}")
print("="*50)
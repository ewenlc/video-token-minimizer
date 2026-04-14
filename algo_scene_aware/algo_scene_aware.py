"""
Stratégie : Scene-aware frame selection
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

THRESHOLD   = 27.0
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "viz_scene_aware.png")

# =====================
# VIDEO INFO
# =====================

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps          = cap.get(cv2.CAP_PROP_FPS)
orig_width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_s         = total_frames_video / video_fps

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
# FRAME SELECTION
# =====================

nb_scenes = len(scenes_frames)

if nb_scenes >= TARGET_FRAMES:
    scenes_with_len = sorted(
        [(i, s[0], s[1], s[1]-s[0]) for i, s in enumerate(scenes_frames)],
        key=lambda x: x[3],
        reverse=True
    )[:TARGET_FRAMES]

    selected_frames = sorted([
        s[1] + (s[2] - s[1]) // 2
        for s in scenes_with_len
    ])
else:
    frames_per_scene = [1] * nb_scenes
    remaining = TARGET_FRAMES - nb_scenes

    scene_lengths = [s[1] - s[0] for s in scenes_frames]

    for _ in range(remaining):
        scores = [
            scene_lengths[i] / (frames_per_scene[i] + 1)
            for i in range(nb_scenes)
        ]
        best = scores.index(max(scores))
        frames_per_scene[best] += 1

    selected_frames = []

    for n, (start, end) in zip(frames_per_scene, scenes_frames):
        if n == 1:
            selected_frames.append((start + end) // 2)
        else:
            indices = np.linspace(start, end, n, dtype=int).tolist()
            selected_frames.extend(indices)

    selected_frames = sorted(selected_frames)

print(f"Selected frames: {len(selected_frames)}")

# =====================
# EXTRACTION + RESIZE
# =====================

extracted = []

for i, frame_num in enumerate(selected_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        continue

    # 🔥 CONTRAINTE GLOBALE
    if TARGET_WIDTH and TARGET_HEIGHT:
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.jpg")
    cv2.imwrite(path, frame)

    extracted.append((frame_num, frame))

cap.release()

# =====================
# VISUALIZATION
# =====================

cols = 4
rows = math.ceil(len(extracted) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
fig.suptitle(
    f"Scene-aware — {len(extracted)} frames / {len(scenes_frames)} scenes",
    fontsize=12
)

axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, (frame_num, frame) in enumerate(extracted):
    axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"{frame_num}")
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
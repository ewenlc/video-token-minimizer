"""
Visualisation de l'espace sémantique CLIP
------------------------------------------
Projette les embeddings CLIP en 2D via UMAP et visualise :
- Chaque point = une frame échantillonnée
- Couleur = cluster KMeans
- Étoile = frame sélectionnée (plus proche du centroïde)
- Taille du point = position temporelle (grand = fin de vidéo)

Dépendances :
    pip install torch transformers scikit-learn umap-learn opencv-python matplotlib numpy Pillow
"""

import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")

import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel
import umap

# =====================
# CONFIGURATION
# =====================

VIDEO_PATH = "../video_test.mp4"
TARGET_FRAMES = 16
NUM_SAMPLED   = 128
BATCH_SIZE    = 16
CLIP_MODEL    = "openai/clip-vit-large-patch14"
OUTPUT_PLOT   = "semantic_space_viz.png"

# =====================
# 1. INFOS VIDÉO + ÉCHANTILLONNAGE
# =====================

print("="*60)
print(f"Vidéo              : {VIDEO_PATH}")
print(f"Frames cibles      : {TARGET_FRAMES}")
print(f"Frames échantillon : {NUM_SAMPLED}")
print(f"Modèle CLIP        : {CLIP_MODEL}")
print("="*60)

cap                = cv2.VideoCapture(VIDEO_PATH)
total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps          = cap.get(cv2.CAP_PROP_FPS)
duration_s         = total_frames_video / video_fps

print(f"\nInfos vidéo :")
print(f"  Total frames    : {total_frames_video}")
print(f"  FPS             : {video_fps:.2f}")
print(f"  Durée           : {duration_s:.1f}s ({duration_s/60:.1f} min)")

num_sampled_actual = min(NUM_SAMPLED, total_frames_video)
sample_indices     = np.linspace(0, total_frames_video - 1, num_sampled_actual, dtype=int).tolist()

print(f"\nÉchantillonnage de {num_sampled_actual} frames...")
frames_pil    = []
frame_indices = []

for idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))
        frame_indices.append(idx)

cap.release()
print(f"  ✓ {len(frames_pil)} frames lues")

# =====================
# 2. ENCODAGE CLIP
# =====================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nChargement de CLIP ({CLIP_MODEL}) sur {device}...")

clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()

print(f"Encodage des {len(frames_pil)} frames...")
embeddings = []

for i in range(0, len(frames_pil), BATCH_SIZE):
    batch  = frames_pil[i:i + BATCH_SIZE]
    inputs = clip_processor(images=batch, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        # get_image_features peut retourner un objet ou un tenseur selon la version
        if not torch.is_tensor(feats):
            feats = feats.image_embeds if hasattr(feats, "image_embeds") else feats.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1)

    embeddings.append(feats.cpu().numpy())
    print(f"  Batch {i//BATCH_SIZE + 1}/{math.ceil(len(frames_pil)/BATCH_SIZE)} encodé")

embeddings = np.concatenate(embeddings, axis=0)
print(f"  ✓ Embeddings shape : {embeddings.shape}")

# =====================
# 3. KMEANS
# =====================

k      = min(TARGET_FRAMES, len(frames_pil))
print(f"\nClustering KMeans (k={k})...")
kmeans  = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels  = kmeans.fit_predict(embeddings)
centers = kmeans.cluster_centers_

# Identifier les frames sélectionnées (plus proches du centroïde)
selected_global_indices = []
for i in range(k):
    cluster_indices = np.where(labels == i)[0]
    if len(cluster_indices) == 0:
        continue
    cluster_embeddings = embeddings[cluster_indices]
    distances          = np.linalg.norm(cluster_embeddings - centers[i], axis=1)
    best_local_idx     = np.argmin(distances)
    best_global_idx    = cluster_indices[best_local_idx]
    selected_global_indices.append(best_global_idx)

print(f"  ✓ {len(selected_global_indices)} frames sélectionnées")

# =====================
# 4. PROJECTION UMAP 2D
# =====================

print(f"\nProjection UMAP en 2D...")
reducer    = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embeddings_2d = reducer.fit_transform(embeddings)
print(f"  ✓ Projection shape : {embeddings_2d.shape}")

# =====================
# 5. VISUALISATION
# =====================

print(f"\nGénération de la visualisation → '{OUTPUT_PLOT}'...")

# Palette de couleurs — une couleur par cluster
cmap    = plt.cm.get_cmap("tab20", k)
colors  = [cmap(i) for i in range(k)]

# Taille des points proportionnelle à la position temporelle
# petit = début de vidéo, grand = fin de vidéo
frame_indices_arr = np.array(frame_indices)
sizes = 30 + 120 * (frame_indices_arr / frame_indices_arr.max())

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    f"Espace sémantique CLIP — {len(frames_pil)} frames → {k} clusters KMeans\n"
    f"Vidéo : {VIDEO_PATH} ({duration_s:.0f}s)  |  Projection UMAP 2D",
    fontsize=13
)

# ---- Graphe gauche : colorié par cluster ----
ax1 = axes[0]
ax1.set_title("Colorié par cluster KMeans", fontsize=11)

is_selected = np.zeros(len(frames_pil), dtype=bool)
is_selected[selected_global_indices] = True

# Points non sélectionnés
for i in range(len(embeddings_2d)):
    if not is_selected[i]:
        ax1.scatter(
            embeddings_2d[i, 0], embeddings_2d[i, 1],
            color=colors[labels[i]],
            s=sizes[i],
            alpha=0.6,
            edgecolors="none"
        )

# Points sélectionnés (étoiles par dessus)
for idx in selected_global_indices:
    ax1.scatter(
        embeddings_2d[idx, 0], embeddings_2d[idx, 1],
        color=colors[labels[idx]],
        s=250,
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label=f"Cluster {labels[idx]+1}"
    )

# Légende clusters
legend_patches = [
    mpatches.Patch(color=colors[i], label=f"Cluster {i+1}")
    for i in range(k)
]
ax1.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=1)
ax1.set_xlabel("UMAP dim 1")
ax1.set_ylabel("UMAP dim 2")
ax1.grid(True, alpha=0.2)

# Annotation : étoile = frame sélectionnée
ax1.scatter([], [], marker="*", color="gray", s=200, edgecolors="black", linewidths=0.8, label="Frame sélectionnée")

# ---- Graphe droit : colorié par position temporelle ----
ax2 = axes[1]
ax2.set_title("Colorié par position temporelle", fontsize=11)

sc = ax2.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1],
    c=frame_indices_arr,
    cmap="plasma",
    s=sizes,
    alpha=0.7,
    edgecolors="none"
)

# Étoiles des frames sélectionnées
for idx in selected_global_indices:
    ax2.scatter(
        embeddings_2d[idx, 0], embeddings_2d[idx, 1],
        c=[frame_indices[idx]],
        cmap="plasma",
        vmin=frame_indices_arr.min(),
        vmax=frame_indices_arr.max(),
        s=250,
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        zorder=5
    )
    # Annoter avec le timestamp
    t = frame_indices[idx] / video_fps
    ax2.annotate(
        f"{t:.0f}s",
        (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
        fontsize=7,
        xytext=(4, 4),
        textcoords="offset points"
    )

cbar = plt.colorbar(sc, ax=ax2)
cbar.set_label("Numéro de frame (début → fin)", fontsize=9)
ax2.set_xlabel("UMAP dim 1")
ax2.set_ylabel("UMAP dim 2")
ax2.grid(True, alpha=0.2)

# Note sur la taille des points
fig.text(
    0.5, 0.01,
    "Taille des points proportionnelle à la position temporelle (petit=début, grand=fin)  |  "
    "★ = frame sélectionnée (plus proche du centroïde)",
    ha="center", fontsize=9, color="gray"
)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
print(f"✓ Visualisation sauvegardée : '{OUTPUT_PLOT}'")

print("\n" + "="*60)
print("RÉSUMÉ")
print("="*60)
print(f"  Frames échantillonnées   : {num_sampled_actual}")
print(f"  Embeddings shape         : {embeddings.shape}")
print(f"  Clusters KMeans          : {k}")
print(f"  Frames sélectionnées     : {len(selected_global_indices)}")
print(f"  Projection               : UMAP 2D")
print(f"  Visualisation            : {OUTPUT_PLOT}")
print("="*60)

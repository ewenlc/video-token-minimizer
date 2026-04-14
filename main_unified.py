import os
import cv2
import subprocess
import math
import argparse

# ====================================================================
# CONFIGURATION GLOBALE
# ====================================================================
TOKEN_BUDGET = 5350             # Budget max pour A100 80Go (batch 16 effectif)
VIDEO_INPUT  = "video_test.mp4" # Ta vidéo source
BASE_OUTPUT  = "dataset_lora"   # Dossier racine pour l'entraînement
PATCH_SIZE   = 28               # Constante Qwen2-VL

# Dossier final pour les images
FRAMES_OUTPUT = os.path.join(BASE_OUTPUT, "frames")

def get_video_dims(path):
    """Récupère les dimensions réelles de la vidéo."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def calculate_n_frames(w, h, budget):
    """Calcule le nombre de frames max pour rester dans le budget."""
    # Qwen2-VL arrondit les dimensions au multiple de 28
    w_28 = round(w / PATCH_SIZE) * PATCH_SIZE
    h_28 = round(h / PATCH_SIZE) * PATCH_SIZE
    
    tokens_per_frame = (w_28 // PATCH_SIZE) * (h_28 // PATCH_SIZE)
    max_frames = budget // tokens_per_frame
    
    return max_frames, tokens_per_frame

def run_strategy(strategy_name):
    # 1. Analyse de la vidéo
    w, h = get_video_dims(VIDEO_INPUT)
    n_frames, tpf = calculate_n_frames(w, h, TOKEN_BUDGET)
    
    print("="*60)
    print(f"ORCHESTRATEUR UNIFIÉ - STRATÉGIE : {strategy_name.upper()}")
    print("="*60)
    print(f"Vidéo source      : {VIDEO_INPUT} ({w}x{h})")
    print(f"Tokens/frame      : {tpf}")
    print(f"Budget total      : {TOKEN_BUDGET} tokens")
    print(f"Frames autorisées : {n_frames}")
    print(f"Destination       : {FRAMES_OUTPUT}")
    print("-"*60)

    if not os.path.exists(FRAMES_OUTPUT):
        os.makedirs(FRAMES_OUTPUT, exist_ok=True)

    # 2. Construction de la commande pour appeler tes scripts
    # On prépare les arguments que tes scripts vont recevoir via argparse
    cmd = ["python"]

    if strategy_name == "temporal_uniform":
        cmd += ["algo_uniform/algo_uniform.py", 
                "--video", VIDEO_INPUT, 
                "--frames", str(n_frames), 
                "--output", FRAMES_OUTPUT]

    elif strategy_name == "scene_aware":
        cmd += ["algo_scene_aware/algo_scene_aware.py", 
                "--video", VIDEO_INPUT, 
                "--frames", str(n_frames), 
                "--output", FRAMES_OUTPUT]

    elif strategy_name == "semantic_keyframe":
        cmd += ["algo_semantic_keyframe/algo_semantic_keyframe.py", 
                "--video", VIDEO_INPUT, 
                "--frames", str(n_frames), 
                "--output", FRAMES_OUTPUT]

    elif strategy_name == "spatial_uniform":
        # Pour cette stratégie, on force une résolution plus basse (ex: 476x280)
        # pour gagner en nombre de frames
        target_w, target_h = 476, 280
        cmd += ["algo_spatial_uniform/algo_spatial_uniform.py", 
                "--video", VIDEO_INPUT, 
                "--width", str(target_w), 
                "--height", str(target_h), 
                "--budget", str(TOKEN_BUDGET),
                "--output", FRAMES_OUTPUT]

    elif strategy_name == "semantic_spatial_crop":
        # YOLO gère son budget dynamiquement
        cmd += ["algo_semantic_spatial_crop/algo_semantic_spatial_crop.py", 
                "--video", VIDEO_INPUT, 
                "--budget", str(TOKEN_BUDGET), 
                "--output", FRAMES_OUTPUT]

    elif strategy_name == "mixte_scene_and_semantic":
        cmd += ["algo_mixte_scene_and_semantic/algo_mixte_scene_and_semantic.py", 
                "--video", VIDEO_INPUT, 
                "--frames", str(n_frames), 
                "--output", FRAMES_OUTPUT]

    else:
        print(f"Erreur : Stratégie '{strategy_name}' inconnue.")
        return

    # 3. Exécution du script cible
    print(f"Exécution de l'algorithme...")
    try:
        subprocess.run(cmd, check=True)
        print("-"*60)
        print(f"SUCCÈS : Les images sont prêtes dans {FRAMES_OUTPUT}")
    except subprocess.CalledProcessError as e:
        print(f"ERREUR lors de l'exécution du script : {e}")

# ====================================================================
# POINT D'ENTRÉE
# ====================================================================
if __name__ == "__main__":
    # Choix possibles : 
    # "temporal_uniform", "scene_aware", "semantic_keyframe", 
    # "spatial_uniform", "semantic_spatial_crop", "mixte_scene_and_semantic"
    
    STRATEGIE_CHOISIE = "semantic_spatial_crop"  # Change cette variable pour tester différentes stratégies
    
    run_strategy(STRATEGIE_CHOISIE)
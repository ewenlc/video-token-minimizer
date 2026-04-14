# Video Data Reduction Toolkit

A unified library of data preparation strategies designed for training Vision-Language Models (VLM) like Qwen2-VL-7B.

The goal is to maximize visual information density while strictly adhering to a token budget, preventing "Out of Memory" (OOM) errors in VRAM-constrained environments (e.g., RTX 3090/4090, A100).

---

## Project Architecture

Everything is managed by the central orchestrator `main_unified.py`, which triggers the specific algorithms:

- algo_uniform/  
  Standard temporal sampling at fixed intervals.

- algo_scene_aware/  
  Scene-cut detection to ensure each shot is represented.

- algo_semantic_keyframe/  
  CLIP embeddings + K-Means clustering to remove visual redundancy.

- algo_semantic_spatial_crop/  
  YOLOv8-based detection to crop and focus on specific subjects (e.g., people).

- algo_spatial_uniform/  
  Dynamic resolution adjustment to fit more frames into the budget.

- algo_mixte_scene_and_semantic/  
  A hybrid approach combining scene detection and semantic filtering.

---

## Installation

# Core dependencies 
```bash
pip install torch torchvision ultralytics transformers scenedetect opencv-python scikit-learn matplotlib pillow
```
---

## How to use

All configurations are handled within `main_unified.py`.

Simply modify the `STRATEGIE_CHOISIE` variable to test different approaches:

# Inside main_unified.py 
```bash
STRATEGIE_CHOISIE = "mixte_scene_and_semantic"  
TOKEN_BUDGET = 5350  # Adjust based on your GPU VRAM  
```

Run the extraction:
```bash
python main_unified.py
```

---

## Token Calculation (Qwen2-VL Specific)

The toolkit automatically calculates the maximum number of allowed frames (N) based on the patch-grid formula:

$$
Tokens = N \times \left(\frac{Width}{28} \times \frac{Height}{28}\right)
$$

Note:  
- Dimensions are automatically rounded to the nearest multiple of 28  
- This matches Qwen2-VL preprocessing requirements  

---

## Strategy Comparison

Strategy | Strength | Ideal Use Case  
Uniform | Simple & Fast | Videos with consistent motion  
Scene Aware | Accuracy | Edited videos or clips with many cuts  
Semantic | Diversity | Removing near-identical frames  
Spatial Crop | Precision | Fine-tuning on specific actions/objects  
Mixte (Hybrid) | Balanced | General-purpose optimization  
"""

# SaliencyMap: Modeling Human Visual Attention for Ad Creative Optimization


---
 
## How It Works?
 
Given any advertising image, this system outputs a **visual saliency heatmap** with a coloured overlay showing where human attention is predicted to concentrate. Red/warm zones indicate high fixation probability; blue/cool zones indicate low attention.
 
This tool is designed for marketing teams, creative directors, and performance marketers who want to evaluate ad design decisions *before* spending budget on live traffic.
 
---
[Live Demo]

---
 
## Model Architecture
 
The model is a **VGG-16 encoder + custom decoder** trained end-to-end on the SALICON eye-tracking dataset:
 
```
Input (224×224×3)
    ↓
VGG-16 Backbone  [frozen, ImageNet weights]
    ↓  (7×7×512 feature map)
Conv2D(512) + UpSampling  →  14×14
Conv2D(256) + UpSampling  →  28×28
Conv2D(128) + UpSampling  →  56×56
Conv2D(64)  + UpSampling  →  112×112
Conv2D(32)  + UpSampling  →  224×224
Conv2D(1, sigmoid)
    ↓
Saliency Map  (224×224×1)  — probability in [0,1]

Dataset: [SALICON](https://www.salicon.net/)

---
 
## Training
 
Training was performed in **three phases** on Google Colab to handle session limits:
 
| Phase | Epochs | `initial_epoch` | Description |
|---|---|---|---|
| Phase 1 | 1 – 10 | 0 | Initial decoder training from random weights |
| Phase 2 | 5 – 10 | 4 | Resume from best Phase 1 checkpoint |
| Phase 3 | 9 – 10 | 8 | Final convergence pass |
 
**Config:**
- Optimizer: `Adam(lr=1e-5)`
- Loss: `MSE` (pixel-wise, continuous maps)
- Augmentation: rotation ±15°, horizontal flip
- Batch size: 16
- Checkpoint: `save_best_only=True`, monitor `loss`
 
---

## Evaluation Metrics
 
| Metric | Description |
|---|---|
| **KL-Divergence** | Distribution distance between predicted and GT fixation maps. ↓ Lower is better. |
| **AUC-Judd** | ROC-AUC with per-image adaptive threshold. ↑ Higher is better. |
 

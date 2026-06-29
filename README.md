# 🧠 Privacy-Aware Continual Knowledge Distillation for Cross-Domain Medical Image Classification

A deep-learning system that learns two different medical-imaging tasks **one after another** — without revisiting old data — while staying small enough to deploy and being **audited for privacy leakage**.
This project provides a **comparative study of continual-learning strategies (naive, EWC, LwF, BatchNorm-freezing)** combined with **knowledge distillation**, across **two teachers and two students**, plus a **membership-inference + differential-privacy** analysis.


## 👤 Authors

- **Bishozit Chandra Das** — Student ID: **VR544246** *(Team Lead)*
- **Abdullah Al Noman Taki** — Student ID: **VR528988**
- **Shaikh Faisal Rahman** — Student ID: **VR546093**

MSc in **Artificial Intelligence**, Università degli Studi di Verona
🔗 [github.com/BishozitChandraDas/Privacy_Aware_Continual_KD](https://github.com/BishozitChandraDas/Privacy_Aware_Continual_KD)


## 🎯 Project Overview

Hospitals keep changing after a model is deployed: new scanners arrive and new diagnostic tasks appear, but old patient data usually **cannot be stored or replayed** for privacy and storage reasons. Updating a model on a new task without the old data makes it **forget** what it already knew — *catastrophic forgetting*.

This project builds a pipeline where a compact student model learns a sequence of medical-imaging tasks while **retaining prior knowledge**, and adds **privacy as an explicit third evaluation axis** alongside accuracy and retention.


## 🧠 Problem Statement

Train a lightweight model to learn **Task A → Task B sequentially without access to Task A data**, compare continual-learning strategies for mitigating catastrophic forgetting, and quantify (then mitigate) how much the trained model **leaks about its training data**.


## 📊 Datasets

Both datasets are downloaded automatically via `kagglehub` during execution.

| Task | Dataset | Classes | Link |
|------|---------|---------|------|
| **Task A** | Chest X-Ray Pneumonia | 2 (Normal, Pneumonia) | [kaggle.com/.../chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Task B** | Brain Tumor MRI | 4 (glioma, meningioma, notumor, pituitary) | [kaggle.com/.../brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |

The two tasks come from **different modalities (X-ray vs. MRI)**, creating a strong domain shift that makes forgetting especially severe — a demanding test bed.


## 🛠️ Pipeline (4 Stages)

### Stage 1 — Teacher Pre-training (→ Task A)
- A **ResNet-50** (and **ViT-B/16** in the ablation) teacher trained on Task A.
- Produces soft labels for distillation.

### Stage 2 — Knowledge Distillation (Teacher → Student)
- Logit-based KD into a lightweight **multi-head student** (**MobileNetV3** / **EfficientNet-B0**).
- Shared backbone + separate Task-A and Task-B heads.

### Stage 3 — Continual Learning (Student → Task B)
- Adapt the student to Task B **without Task A data**, comparing four strategies:
  - **Naive** fine-tuning (baseline / lower bound)
  - **EWC** — Elastic Weight Consolidation (Fisher-weighted penalty)
  - **LwF** — Learning without Forgetting (self-distillation)
  - **BN-Freeze** — freeze BatchNorm running statistics

### Stage 4 — Privacy (novel contribution)
- **Membership Inference Attack (MIA)** measures training-data leakage.
- **DP-SGD** (via Opacus) mitigates it; we chart the **privacy–utility trade-off** across budgets ε.


## ⚙️ Experimental Setup

**Platform:** Google Colab (T4 GPU) or local machine
**Framework:** PyTorch + Opacus

**Hyperparameters:**
- Image size: 224×224, ImageNet normalization, augmentation (flip, rotation)
- Optimizer: Adam, learning rate 1e-3, batch size 32
- Distillation: temperature T = 3, α = 0.5
- EWC λ = 2000 (selected via ablation), continual epochs configurable


## 📈 Results

### Teacher vs. Distilled Student (Task A)
| Model | Accuracy | AUC |
|-------|----------|-----|
| Teacher (ResNet-50) | 95.79% | 0.991 |
| **Student (MobileNetV3, distilled)** | **97.04%** | **0.994** |

The student **exceeds the teacher** despite being ~5× smaller — distillation transfers knowledge effectively.

### Continual Learning — Task A Retention (%) across all teacher × student pairs
| Teacher → Student | Naive | EWC | LwF | **BN-Freeze** |
|-------------------|-------|-----|-----|---------------|
| ResNet-50 → MobileNetV3 | 89.6 | 93.9 | 75.0 | **100.0** |
| ResNet-50 → EfficientNet-B0 | 41.9 | 95.5 | 76.9 | **100.0** |
| ViT-B/16 → MobileNetV3 | 67.9 | 81.8 | 75.2 | **100.0** |
| ViT-B/16 → EfficientNet-B0 | 27.8 | 59.7 | 92.1 | **100.0** |

→ **Naive forgets the most** (down to 27.8%); **BatchNorm-freezing retains ~100%** in every case while keeping Task-B accuracy high (~87–96%).

### EWC λ Ablation
| λ | Task A Retention (%) | Task B Accuracy |
|---|----------------------|-----------------|
| 0 (naive) | 86.3 | 0.933 |
| 500 | 95.0 | 0.959 |
| **2000** | **95.4** | **0.966** |
| 8000 | 90.6 | 0.931 |

→ Retention peaks near **λ = 2000**; too-large λ over-regularizes and hurts both metrics.

### Privacy–Utility Trade-off
| ε (budget) | Task A Accuracy | MIA AUC |
|------------|-----------------|---------|
| ∞ (no DP) | 0.919 | 0.526 |
| 8 | 0.854 | 0.512 |
| 3 | 0.774 | 0.506 |
| 1 | 0.861 | 0.503 |

→ Membership leakage is already **low** (AUC ≈ 0.53); DP-SGD drives it toward **chance (0.50)** at a measurable accuracy cost.


## 🔍 Key Findings

- A small distilled student can **match or beat its teacher** on the source task.
- **Naive fine-tuning suffers severe catastrophic forgetting** under cross-modality shift.
- **Freezing BatchNorm statistics is the single most effective fix** — near-perfect retention with no extra memory or stored data.
- **EWC** gives strong, tunable retention; the best *algorithmic* strategy can depend on the backbone.
- The distilled, regularized models **leak little membership information**, and **DP-SGD adds a formal guarantee** at an accuracy cost.


## 📁 Repository Structure

```
APAI_Project/
├── config.py              # all settings / hyperparameters
├── main.ipynb             # orchestrates the full pipeline (with outputs)
├── src/
│   ├── data.py            # download, clean, split, loaders
│   ├── models.py          # teachers + multi-head students
│   ├── engine.py          # train / evaluate / metrics / checkpoints
│   ├── distillation.py    # logit-based KD
│   ├── continual.py       # naive / EWC / LwF / BN-freeze
│   ├── privacy.py         # MIA + DP-SGD (GroupNorm DPNet)
│   ├── experiments.py     # all teacher×student pairs + λ sweep
│   └── plots.py           # all figures
└── results/
    ├── *.csv              # all numerical results
    └── figures/           # ROC, confusion, retention, privacy, etc.
```

> **Note:** Pre-trained checkpoints (`checkpoints/`) are **not included** due to size (~800 MB). Running `main.ipynb` top-to-bottom automatically downloads the datasets and trains all models, saving checkpoints locally. All reported results and figures are already in `results/`.


## 📋 Requirements

### Environment
- **Platform:** Google Colab (recommended, T4 GPU) or local machine
- **Python:** 3.10+
- **Hardware:** GPU recommended (CPU works but is slow)

### Dependencies
```bash
pip install torch torchvision
pip install opacus
pip install kagglehub
pip install scikit-learn
pip install matplotlib seaborn
pip install numpy pandas
```


## 🔧 Setup & ▶️ How to Run

### Google Colab (Recommended)
1. Upload the `APAI_Project` folder to your Google Drive (`MyDrive/APAI_Project/`).
2. Open `main.ipynb` in Colab and set **Runtime → T4 GPU**.
3. Run all cells sequentially.
4. Datasets download automatically via `kagglehub`; models train and checkpoint themselves.

### Local Environment
1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies above.
4. Run `main.ipynb` (or the scripts) top-to-bottom.

> Each stage is **checkpoint-based** — if a checkpoint exists it is loaded, otherwise it is trained and saved, so runs are resumable after a Colab timeout.


## 📌 Limitations

- Evaluation covers **two sequential tasks** and a multi-head design.
- The DP study uses a compact **GroupNorm** network (DP-SGD is incompatible with BatchNorm / residual in-place ops).
- The membership-inference attack is the **simplest loss-based** attacker.
- Training subsets were capped for tractability; full-data runs would improve absolute scores.


## 🎓 Academic Context

Developed as the team project for the **Advanced Programming for AI (APAI)** course, MSc in Artificial Intelligence, University of Verona. The work focuses on:
- Knowledge distillation and continual learning
- Architectural vs. algorithmic mitigation of catastrophic forgetting
- Privacy auditing (MIA) and formal privacy guarantees (DP-SGD)


## 📄 Notes
- Intended for educational and research purposes.
- Not optimized for clinical or production use.

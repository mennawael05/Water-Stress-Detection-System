# 💧 Water Stress Prediction — Model Comparison

>What your project does and what problem it solves
Groundwater is invisible — you can't see it drying up the way you can see a river. Egypt and the wider MENA region face severe groundwater depletion, but measuring it requires expensive wells or waiting for satellite data that arrives months late. Your project builds a model that predicts groundwater storage levels using data that is already freely available every month, so you can estimate what's happening underground without needing to drill anything.
The output is a number called GRACE LWE (Land Water Equivalent), measured in centimetres. A negative number means groundwater is depleting. Your model learns to predict that number for any location and month.

---

## 🌍 Problem Statement

Water scarcity is one of the most critical global challenges. This project addresses **early prediction of water-stressed regions** using satellite imagery and hydrological data — enabling proactive intervention *before* a water crisis materializes.

### Data Sources
| Source | Features | Description |
|--------|----------|-------------|
| **GLDAS** (Global Land Data Assimilation System) | 9 features | Soil moisture (3 layers), root moisture, evapotranspiration, runoff, rainfall, temperature |
| **Sentinel-2 Satellite Indices** | 5 features | NDVI, NDWI, MNDWI, BSI, NDSI |
| **GRACE LWE** | Target (regression) | Groundwater storage anomaly in cm |

### Input Tensor Shape
```
(N, T, H, W, C)
  N = number of samples
  T = timesteps (5 or 12 months)
  H × W = spatial grid (4×4 or 1×1)
  C = features (7 or 14)
```

---

## 🤖 Models Overview

Five deep learning architectures were developed and compared for water stress prediction:

| # | Model | File | Task | Input |
|---|-------|------|------|-------|
| 1 | **CNN + LSTM** | `cnn_lstm.ipynb` | Binary Classification | `(N, T, H, W, 7)` |
| 2 | **CNN + Transformer** | `CNN___Transformer.ipynb` | Binary Classification | `(N, T, H, W, 7)` |
| 3 | **ViT** (Vision Transformer) | `VIT.ipynb` | Binary Classification | `(N, T, H, W, 7)` |
| 4 | **CNN + RNN** (GLDAS+Satellite) | `CNN_RNN_GLDAS_Satellite.ipynb` | Regression | `(N, 24, 15)` |
| 5 | **ViT + Bi-LSTM** | `ViT_LSTM_WaterStress_SequenceInput_1__1_.ipynb` | Regression | `(N, 12, 1, 1, 14)` |

---

## 🏗️ Architecture Details

### 1. CNN + LSTM
```
Input (N, T, H, W, 7)
   └─► SpatialCNN per timestep
         Conv2d(7→16) + BN + ReLU
         Conv2d(16→32) + BN + ReLU
         AdaptiveAvgPool2d(1)
         Linear(32→32) + Dropout(0.5)
   └─► LSTM(32, hidden=64, layers=1)
   └─► Linear(64→32) → Linear(32→1) → Sigmoid
```
- **Task:** Binary classification (Stressed / Not-Stressed)
- **Loss:** BCE Loss
- **Optimizer:** Adam (lr=1e-3), ReduceLROnPlateau
- **Metrics:** Accuracy, ROC-AUC, F1, Confusion Matrix

---

### 2. CNN + Transformer
```
Input (N, T, H, W, 7)
   └─► CNN per timestep
         Conv2d(7→16) + ReLU
         Conv2d(16→8) + ReLU
         Flatten → feature_dim = 8×4×4 = 128
   └─► TransformerEncoder(d_model=128, nhead=4, layers=1, dropout=0.5)
   └─► Linear(128→32) → Dropout(0.5) → Linear(32→1) → Sigmoid
```
- **Task:** Binary classification
- **Loss:** BCE Loss
- **Optimizer:** Adam (lr=0.0005, weight_decay=1e-4), ReduceLROnPlateau
- **Early Stopping:** patience=7
- **Metrics:** Accuracy, ROC-AUC, F1

---

### 3. ViT (Vision Transformer)
```
Input (N, T=5, H=4, W=4, C=7)
   └─► Patchify: patch_size=2 → 4 patches per timestep
   └─► Linear Patch Embedding → d_model=64
   └─► [CLS token] + Positional Embedding + Temporal Embedding
   └─► TransformerEncoder(d_model=64, heads=4, layers=4, dropout=0.15)
   └─► LayerNorm → Linear(64→32, GELU) → Linear(32→2)
```
- **Task:** Binary classification (2-class softmax)
- **Loss:** CrossEntropyLoss (class-weighted + label smoothing=0.1)
- **Optimizer:** AdamW (lr=2e-4, weight_decay=5e-3)
- **LR Schedule:** Warmup (10 epochs) → CosineAnnealing
- **Augmentation:** Noise injection, channel dropout, spatial flips, MixUp
- **Early Stopping:** patience=35

---

### 4. CNN + RNN (GLDAS + Satellite)
```
Input (N, seq_len=24, features=15)
   └─► Conv1d blocks (1D temporal CNN)
         [15→32→64] Conv1d + BN + ReLU + Dropout
   └─► RNN(hidden=128, layers=2, dropout=0.3)
   └─► Linear(128→64) → ReLU → Dropout → Linear(64→1)
```
- **Task:** Regression (predicting water stress index)
- **Loss:** MSE Loss
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Metrics:** RMSE, MAE, R²
- **Sequence Length:** 24 months sliding window

---

### 5. ViT + Bi-LSTM (Most Advanced)
```
Input (N, T=12, H=1, W=1, C=14)
   └─► ViT Encoder per timestep (patch=1×1, acts as feature projector)
         Linear(14→128) + Positional Embedding
         TransformerEncoder → embed_t (dim=128)
   └─► 2-layer Bi-LSTM(128→256 bidirectional)
   └─► Temporal Attention → weighted context vector
   └─► Regression Head → GRACE LWE anomaly (cm)
```
- **Task:** Regression (GRACE groundwater storage anomaly)
- **Loss:** Huber Loss (robust to outliers)
- **Optimizer:** AdamW + CosineAnnealing LR
- **Metrics:** RMSE, MAE, R²
- **Split Strategy:** Spatial split (no location leakage) — 70/15/15
- **Features:** 9 GLDAS + 5 Satellite indices = 14 features

---

## ⚖️ Model Comparison

| Criterion | CNN+LSTM | CNN+Transformer | ViT | CNN+RNN | ViT+BiLSTM |
|-----------|----------|-----------------|-----|---------|------------|
| **Task** | Classification | Classification | Classification | Regression | Regression |
| **Spatial Modeling** | ✅ Conv2D | ✅ Conv2D | ✅ Patch Attention | ❌ Conv1D (temporal) | ✅ ViT patches |
| **Temporal Modeling** | ✅ LSTM | ✅ Transformer | ⚠️ Implicit (positional) | ✅ RNN | ✅ Bi-LSTM + Attention |
| **Data Sources** | Synthetic tensors | Synthetic tensors | Synthetic tensors | GLDAS + Satellite | GLDAS + Satellite |
| **Sequence Length** | 5 timesteps | 5 timesteps | 5 timesteps | 24 months | 12 months |
| **Features** | 7 | 7 | 7 | 15 | 14 |
| **Augmentation** | ❌ | ❌ | ✅ MixUp + flips | ❌ | ❌ |
| **Class Imbalance Handling** | ❌ | ❌ | ✅ Weighted loss | N/A | N/A |
| **Complexity** | Medium | Medium | High | Medium | Very High |
| **Output** | Binary label | Binary label | Binary label | Continuous index | GRACE LWE (cm) |
| **Leakage Protection** | Stratified split | Stratified split | Stratified split | Random split | ✅ Spatial split |

---

## 🏆 Recommended Model for Production

### 🥇 Best for Early Warning Systems: **ViT + Bi-LSTM**

**Why:**
- Uses **real-world data** (GLDAS + GRACE + Sentinel-2) — not synthetic
- **Spatial split** prevents data leakage between geographic locations
- **Bi-LSTM + Attention** captures long-range temporal dependencies (12-month window)
- **Huber Loss** is robust to outliers common in hydrology data
- Predicts **GRACE LWE anomaly** — a direct physical indicator of groundwater storage changes
- Designed explicitly for the water scarcity prediction problem

### 🥈 Best Classifier: **ViT**

**Why:**
- Most sophisticated training setup (MixUp, data augmentation, class-weighted loss)
- Handles class imbalance explicitly
- Self-attention over spatial patches captures complex drought patterns
- Temporal + positional embeddings model spatio-temporal context

### 🥉 Best Lightweight Option: **CNN + LSTM**

**Why:**
- Simple, interpretable, fast to train
- BatchNorm + Dropout for stable training
- Good baseline for binary water stress classification

---

## 📁 Repository Structure

```
├── cnn_lstm.ipynb                         # CNN + LSTM classifier
├── CNN___Transformer.ipynb                # CNN + Transformer classifier
├── VIT.ipynb                              # Vision Transformer classifier
├── CNN_RNN_GLDAS_Satellite.ipynb          # CNN + RNN regressor (GLDAS + Satellite)
├── ViT_LSTM_WaterStress_SequenceInput.ipynb  # ViT + Bi-LSTM regressor (most advanced)
├── data/
│   ├── water_stress_tensors_generate.npy  # Synthetic spatial tensors (classifiers)
│   ├── water_stress_labels_generate.npy   # Binary labels (classifiers)
│   ├── grace_gldas_trainable.parquet      # GLDAS + GRACE real data
│   ├── satellite_indices_s2_period.parquet # Sentinel-2 indices
│   └── lstm_feature_names.csv             # GLDAS feature names
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn einops torchmetrics
```

### Running a Model
```python
# Example: CNN + LSTM
# 1. Load data
X = np.load('data/water_stress_tensors_generate.npy')
y = np.load('data/water_stress_labels_generate.npy').astype(np.float32)

# 2. Open notebook
jupyter notebook cnn_lstm.ipynb

# 3. Run all cells (Google Colab recommended for GPU)
```

### For Real-World Data (ViT + Bi-LSTM)
```python
# Requires:
# - grace_gldas_trainable.parquet  (GLDAS + GRACE LWE)
# - satellite_indices_s2_period.parquet  (Sentinel-2)

jupyter notebook "ViT_LSTM_WaterStress_SequenceInput_1__1_.ipynb"
```

---

## 📊 Evaluation Metrics

| Task | Metrics Used |
|------|-------------|
| Classification | Accuracy, ROC-AUC, F1-Score, Confusion Matrix, Precision, Recall |
| Regression | RMSE, MAE, R² (coefficient of determination) |

---

## 🔬 Key Technical Choices

| Decision | Rationale |
|----------|-----------|
| Spatial split (ViT+BiLSTM) | Prevents geographic leakage — adjacent grid cells share climate patterns |
| Huber Loss for regression | Robust to extreme drought/flood outliers in GRACE data |
| MixUp augmentation (ViT) | Improves generalization with limited labeled stress events |
| Sliding window sequences | Captures seasonal patterns and gradual moisture depletion trends |
| Bi-directional LSTM | Learns both past trends and future context in monthly sequences |
| Temporal Attention | Weights the most informative months in the 12-month window |

---

## 🌐 Real-World Impact

Early detection of water stress enables:
- **Agricultural planning**: Optimize irrigation before crop failure
- **Government policy**: Allocate water resources proactively  
- **Humanitarian response**: Pre-position aid before drought emergencies
- **Infrastructure**: Early reservoir management decisions

---



---

*Built with PyTorch | Data: NASA GLDAS + GRACE + ESA Sentinel-2*

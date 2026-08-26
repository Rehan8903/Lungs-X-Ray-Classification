# Pneumonia X-Ray Classification — MLOps Project

An end-to-end MLOps pipeline that classifies chest X-ray images as **NORMAL** or **PNEUMONIA**, using a fine-tuned ResNet18 (PyTorch). Built on the Kaggle [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset.

## Project Pipeline

```
Project Structure → Dataset → EDA → CNN Experiments → Data Ingestion →
Image Preprocessing → Model Training → Evaluation → Model Registry →
DVC Pipeline → FastAPI → Frontend → CI/CD → Docker → Deployment
```

**Status:** Data Ingestion through Frontend complete. CI/CD, Docker, and Deployment remain.

## What's Built So Far

### 1. EDA & CNN Experiments (`notebooks/`)
- `01_eda.ipynb` — class balance, image dimensions/color mode, pixel intensity distributions, corrupted file checks
- `02_cnn_experiments.ipynb` — compared a from-scratch baseline CNN vs. a fine-tuned ResNet18. **ResNet18 won** (ROC-AUC 0.959 vs 0.933 on Colab)

### 2. MLOps Components (`src/components/`)
| File | Purpose |
|---|---|
| `data_ingestion.py` | Downloads dataset via Kaggle API, resolves nested zip structure, re-splits `train/` into a proper train/val split (official `val/` only has 16 images) |
| `data_preprocessing.py` | Builds transforms (resize, grayscale→3-channel, augmentation), datasets, and dataloaders with a weighted sampler for class imbalance |
| `model.py` | Architecture definitions — `BaselineCNN` and `build_resnet18()` |
| `model_trainer.py` | Training loop with early stopping, LR scheduling, and a two-phase frozen→fine-tune process for ResNet18 |
| `model_evaluation.py` | Test-set metrics (accuracy, ROC-AUC, per-class precision/recall/F1), confusion matrix, saved as JSON/PNG artifacts |
| `model_registry.py` | Lightweight file-based model registry — versions models (`v1`, `v2`, ...) and auto-promotes the best one (by ROC-AUC) to "production" |

All components are config-driven via `config/config.yaml` and use shared `src/logger.py` / `src/exception.py` for consistent logging and error handling.

### 3. DVC Pipeline (`dvc.yaml`)
Wires the components into four reproducible stages:
```
data_ingestion → train → evaluate → registry
```
Run the full pipeline with:
```bash
python -m dvc repro
```
DVC skips stages whose inputs haven't changed, and `dvc metrics show` compares runs.

**Current production model:** `v1` — ResNet18, test accuracy 87.3%, ROC-AUC 0.964.

### 4. FastAPI Serving (`app/`)
| File | Purpose |
|---|---|
| `app.py` | FastAPI app — loads the current production model from the registry at startup, serves `/predict`, `/health`, `/model-info`, and the frontend |
| `preprocessing_utility.py` | Shared inference-time preprocessing (single source of truth, matches training transforms exactly) |
| `load_model_test.py` | Standalone sanity check — confirms the production model loads and predicts correctly before starting the API |
| `template/` | Frontend — `index.html`, `style.css`, `script.js` |

**Endpoints:**
- `GET /` — serves the web UI
- `GET /health` — liveness check
- `GET /model-info` — current production model version + metrics
- `POST /predict` — upload an X-ray image, get back prediction + confidence + class probabilities

### 5. Frontend (`app/template/`)
A dark-themed single-page UI: drag-and-drop X-ray upload, live preview, animated confidence bars, and a live model-status badge. Talks directly to the FastAPI backend on the same origin.

## Running It Locally

```bash
# 1. Set Kaggle API credentials (see Kaggle account settings → API)
# Windows: setx KAGGLE_API_TOKEN "your_token_here"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full ML pipeline (ingestion → train → evaluate → registry)
python -m dvc repro

# 4. Start the API + frontend
python -m app.app
```
Then open `http://localhost:8000/` in a browser.

## Known Model Limitations

- **NORMAL-class recall is weaker** (~0.6–0.7) than PNEUMONIA recall (~0.99) — the model over-predicts pneumonia. Reasonable for a screening context (missing real pneumonia is worse than a false alarm), but not yet tuned to reduce false positives.
- **Val/test distribution gap** — validation accuracy during training runs notably higher than final test accuracy, a known characteristic of this specific Kaggle dataset (test set sourced somewhat differently from train).
- This is a portfolio/educational project — **not a diagnostic tool**.

## Remaining Steps

- [ ] CI/CD (GitHub Actions)
- [ ] Docker containerization
- [ ] Deployment
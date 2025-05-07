# Grammar Scoring Engine 

This repository implements a **hybrid grammar-scoring engine** that automatically evaluates spoken English responses on a continuous 0–5 grammar proficiency scale (MOS-based). It combines hand‑crafted linguistic features, classical machine‑learning models, and a fine‑tuned DistilBERT regression to mimic expert human scoring.

---

## 🚀 Quickstart

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/grammar-scoring-engine.git
   cd grammar-scoring-engine
   ```

2. **Download data via Kaggle API**

   ```bash
   mkdir -p data
   kaggle competitions download -c shl-hiring-assessment -p data/
   unzip data/*.zip -d data/
   ```

## 📋 Methodology

1. **Audio Preprocessing**: Resample to 16 kHz, normalize amplitude, trim silence.
2. **ASR Transcription**: Generate raw transcripts using OpenAI Whisper‑base.
3. **Transcript Cleaning**: Lowercase, remove disfluencies (`um`, `uh`), standardize punctuation.
4. **Feature Engineering**:
   * Grammar errors (LanguageTool)
   * Syntactic complexity & POS diversity (spaCy)
   * GEC edit distance & edit rate (T5-based)
5. **Classical ML Ensemble**: RandomForest, LightGBM, Ridge on handcrafted features → `feature_pred`.
6. **Deep Learning**: Fine‑tune DistilBERT for regression → `bert_pred`.
7. **Meta‑Stacking**: Ridge regressor combines `[feature_pred, bert_pred]` → final score.

**Evaluation (5‑fold CV):**  
- RMSE: **0.7190**  
- Pearson Correlation: **0.5873**
  
---
## 🔧 Next Steps

* Incorporate audio-fluency features (pause durations, speaking rate)
* Explore multimodal fusion (wav2vec2 + text features)
* Hyperparameter tuning via Optuna or Bayesian search

---

## 📄 License

MIT License © 2025 Lakshya Singh

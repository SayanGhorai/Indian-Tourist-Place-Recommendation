# 🇮🇳 Indian Tourist Place Recommendation

A hybrid **TF-IDF + SBERT** based recommendation system for Indian tourist places.

Search using natural language queries like:

- temples in varanasi
- beach in goa

The system automatically detects **city + intent** and returns **ranked place-level results** with smart ranking.

---

## ✨ Features

- Strict city filtering (no cross-city results)
- Hybrid search (**TF-IDF + SBERT semantic similarity**)
- Automatic intent tagging (temple, beach, fort, food, etc.)
- Confidence-aware ranking (**rating × review count**)
- Fast CLI interface
- Easy to extend to Streamlit / Web app

---

## 🏗️ Architecture

Dataset → Cleaning → Place Aggregation → TF-IDF + SBERT → Auto Tagging → Hybrid Ranking → Top Results

---

## 🛠️ Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- sentence-transformers
- torch

Install dependencies:

```bash
pip install -r requirements.txt
```

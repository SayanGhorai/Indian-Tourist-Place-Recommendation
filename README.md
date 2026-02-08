# 🇮🇳 Indian Tourist Place Recommendation

A hybrid **TF-IDF + SBERT** based recommendation system for Indian tourist places.

Search using natural language queries like:

- temples in varanasi
- beach in goa

The system automatically detects **city + intent** and returns **ranked place-level results** with smart ranking.

---

## ✨ Features

- Loads and cleans tourist review dataset
- Groups data at place level (City + Place)
- Searches using TF-IDF and SBERT similarity
- Automatically detects place type (temple, beach, fort, food, etc.)
- Returns results only from the requested city
- Ranks places using ratings and number of reviews
- Simple command line interface
- Easy to extend into a web app (Streamlit/Flask)

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

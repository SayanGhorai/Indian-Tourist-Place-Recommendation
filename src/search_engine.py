import os
import re
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util

from src.tagging import (
    TAG_QUERIES,
    CITY_ALIAS_MAP,
    detect_city,
    detect_intent,
    print_diagnostics,
    init_sbert_model
)

# ---------------- Global Objects ----------------
place_reviews = None
tfidf = None
tfidf_matrix = None
model = None
corpus_embeddings = None
tag_embeddings = None
tag_names = None
tag_norm = None


# ---------------- Helpers ----------------
def _norm_city(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _empty_results_df():
    cols = [
        "City", "Place", "avg_rating", "review_count",
        "auto_tag", "auto_tags", "top_keywords",
        "pros", "cons", "recommendation_explanation",
        "final_score"
    ]
    return pd.DataFrame(columns=cols)


# ---------------- Load Backend ----------------
def load_search_backend(
    prepared_csv="data/places_genai_ready.csv",
    load_embeddings=True,
    sbert_model_name="all-MiniLM-L6-v2"
):
    global place_reviews, tfidf, tfidf_matrix
    global model, corpus_embeddings, tag_embeddings
    global tag_names, tag_norm

    if not os.path.exists(prepared_csv):
        raise FileNotFoundError(f"{prepared_csv} not found")

    place_reviews = pd.read_csv(prepared_csv)

    # rebuild full_text if missing
    if "full_text" not in place_reviews.columns:
        place_reviews["full_text"] = (
            place_reviews["City"].fillna("").astype(str) + " " +
            place_reviews["Place"].fillna("").astype(str) + " " +
            place_reviews["recommendation_explanation"].fillna("").astype(str)
        )

    # rebuild City_canon if missing
    if "City_canon" not in place_reviews.columns:
        place_reviews["City_canon"] = (
            place_reviews["City"]
            .fillna("")
            .astype(str)
            .apply(_norm_city)
            .apply(lambda x: CITY_ALIAS_MAP.get(x, x))
        )

    # parse list-like columns
    for col in ["auto_tags", "top_keywords", "pros", "cons"]:
        if col in place_reviews.columns:
            place_reviews[col] = place_reviews[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )

    # TF-IDF
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=70000
    )

    tfidf_matrix = tfidf.fit_transform(
        place_reviews["full_text"].astype(str)
    )

    tag_names = list(TAG_QUERIES.keys())

    if load_embeddings:
        model = init_sbert_model(sbert_model_name)

        corpus_embeddings = model.encode(
            place_reviews["full_text"].tolist(),
            batch_size=64,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        tag_texts = [TAG_QUERIES[t] for t in tag_names]

        tag_embeddings = model.encode(
            tag_texts,
            convert_to_tensor=True
        )

        tag_norm = F.normalize(tag_embeddings, p=2, dim=1)

    print("Search backend loaded")
    print(f"Places: {len(place_reviews)}")
    print(f"TF-IDF shape: {tfidf_matrix.shape}")


# ---------------- Search ----------------
def hybrid_search_with_tags(
    query,
    top_n=5,
    soft_boost=True,
    boost_weight=0.15,
    rating_boost_weight=0.15
):
    global place_reviews, tfidf, tfidf_matrix
    global model, corpus_embeddings
    global tag_names, tag_norm

    if place_reviews is None:
        raise RuntimeError("Run load_search_backend() first")

    n_places = len(place_reviews)

    q_vec = tfidf.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    query_emb = model.encode(query, convert_to_tensor=True)

    sbert_scores = util.cos_sim(
        query_emb,
        corpus_embeddings
    ).cpu().numpy().flatten()

    final_scores = (0.35 * tfidf_scores) + (0.65 * sbert_scores)

    # city detection
    canonical_city_vocab = sorted(
        place_reviews["City_canon"].dropna().unique().tolist()
    )

    detected_city, city_method, _ = detect_city(
        query,
        canonical_city_vocab
    )

    city_mask = np.ones(n_places, dtype=bool)

    if detected_city:
        city_mask = (
            place_reviews["City_canon"]
            .astype(str)
            == detected_city
        ).values

        if city_mask.sum() == 0:
            return _empty_results_df()

    mask = city_mask.copy()

    # intent detection
    detected_tags, _ = detect_intent(
        query,
        model,
        tag_norm,
        tag_names
    )

    # soft boost using auto_tags
    if soft_boost and detected_tags:
        for i in range(n_places):
            tags_here = place_reviews.iloc[i]["auto_tags"]
            overlap = len(set(tags_here) & set(detected_tags))
            final_scores[i] += boost_weight * overlap

    # rating boost
    rating_scores = (
        place_reviews["avg_rating"].fillna(0).values *
        np.log1p(place_reviews["review_count"].fillna(0).values)
    )

    if rating_scores.max() > 0:
        rating_scores = rating_scores / rating_scores.max()

    final_scores += rating_boost_weight * rating_scores

    candidates_idx = np.where(mask)[0]

    if len(candidates_idx) == 0:
        return _empty_results_df()

    candidate_scores = final_scores[candidates_idx]
    order = np.argsort(candidate_scores)[::-1]
    top_idx = candidates_idx[order[:top_n]]

    result_cols = [
        "City", "Place", "avg_rating", "review_count",
        "auto_tag", "auto_tags", "top_keywords",
        "pros", "cons", "recommendation_explanation"
    ]

    result = place_reviews.loc[top_idx, result_cols].copy()
    result["final_score"] = final_scores[top_idx]

    print_diagnostics(
        query=query,
        detected_city=detected_city,
        city_method=city_method,
        detected_tags=detected_tags,
        city_mask_sum=int(city_mask.sum()),
        candidate_count_after_tag_filter=len(candidates_idx)
    )

    return result.reset_index(drop=True)
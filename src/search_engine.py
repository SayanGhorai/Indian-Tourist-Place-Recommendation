import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util

from src.tagging import (
    TAG_QUERIES,
    PHRASE_TO_TAG,
    detect_city,
    detect_intent,
    print_diagnostics,
    init_sbert_model
)

# ---------------- Global Runtime Objects ----------------
place_reviews = None
tfidf = None
tfidf_matrix = None
model = None
corpus_embeddings = None
tag_embeddings = None
tag_names = None
tag_norm = None


# ---------------- Load Runtime Backend ----------------
def load_search_backend(
    prepared_csv="data/places_genai_ready.csv",
    load_embeddings=True,
    sbert_model_name="all-MiniLM-L6-v2"
):
    global place_reviews, tfidf, tfidf_matrix
    global model, corpus_embeddings, tag_embeddings, tag_names, tag_norm

    if not os.path.exists(prepared_csv):
        raise FileNotFoundError(f"{prepared_csv} not found")

    place_reviews = pd.read_csv(prepared_csv)

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

    # SBERT runtime
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
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")


# ---------------- Empty Results ----------------
def _empty_results_df():
    cols = [
        "City", "Place", "avg_rating", "review_count",
        "auto_tag", "auto_tags", "top_keywords",
        "pros", "cons", "recommendation_explanation",
        "final_score"
    ]
    return pd.DataFrame(columns=cols)


# ---------------- Main Search ----------------
def hybrid_search_with_tags(
    query,
    top_n=5,
    soft_boost=True,
    boost_weight=0.15,
    rating_boost_weight=0.15
):
    global place_reviews, tfidf, tfidf_matrix
    global model, corpus_embeddings, tag_embeddings, tag_names, tag_norm

    if place_reviews is None:
        raise RuntimeError("Run load_search_backend() first")

    n_places = len(place_reviews)

    # TF-IDF scores
    q_vec = tfidf.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # SBERT scores
    query_emb = model.encode(query, convert_to_tensor=True)

    sbert_scores = util.cos_sim(
        query_emb,
        corpus_embeddings
    ).cpu().numpy().flatten()

    # Hybrid scoring
    final_scores = (0.4 * tfidf_scores) + (0.6 * sbert_scores)

    # ---------------- City Detection ----------------
    canonical_city_vocab = sorted(
        place_reviews["City_canon"].dropna().unique().tolist()
    )

    detected_city, city_method, _ = detect_city(
        query,
        canonical_city_vocab
    )

    city_mask = np.ones(n_places, dtype=bool)

    if detected_city:

        # strict exact match first
        city_mask = (
            place_reviews["City_canon"]
            .fillna("")
            .astype(str)
            == detected_city
        ).values

        # fallback for Goa-like fragmented cities
        if city_mask.sum() == 0:
            city_mask = (
                place_reviews["City_canon"]
                .fillna("")
                .astype(str)
                .str.contains(detected_city, case=False, na=False)
            ).values

        if city_mask.sum() == 0:
            print(f"No results in {detected_city}")
            return _empty_results_df()

    mask = city_mask.copy()

    # ---------------- Intent Detection ----------------
    detected_tags, detect_method = detect_intent(
        query,
        model,
        tag_norm,
        tag_names
    )

    # ---------------- Soft Boost ----------------
    if soft_boost and detected_tags:
        for tag in detected_tags:
            col = f"is_{tag}"
            if col in place_reviews.columns:
                final_scores += (
                    boost_weight *
                    place_reviews[col].astype(float).values
                )

    # ---------------- Rating + Popularity Boost ----------------
    rating_scores = (
        place_reviews["avg_rating"].fillna(0).values *
        np.log1p(place_reviews["review_count"].fillna(0).values)
    )

    if rating_scores.max() > 0:
        rating_scores = rating_scores / rating_scores.max()

    final_scores += rating_boost_weight * rating_scores

    # ---------------- Filter Candidates ----------------
    candidates_idx = np.where(mask)[0]

    if len(candidates_idx) == 0:
        return _empty_results_df()

    candidate_scores = final_scores[candidates_idx]

    order = np.argsort(candidate_scores)[::-1]
    top_idx = candidates_idx[order[:top_n]]

    # ---------------- Final Results ----------------
    result_cols = [
        "City", "Place", "avg_rating", "review_count",
        "auto_tag", "auto_tags", "top_keywords",
        "pros", "cons", "recommendation_explanation"
    ]

    result = place_reviews.loc[top_idx, result_cols].copy()

    result["final_score"] = final_scores[top_idx]

    result = result.reset_index(drop=True)

    # Diagnostics
    print_diagnostics(
        query=query,
        detected_city=detected_city,
        city_method=city_method,
        detected_tags=detected_tags,
        city_mask_sum=int(city_mask.sum()),
        candidate_count_after_tag_filter=len(candidates_idx)
    )

    return result
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from src.tagging import TAG_QUERIES, detect_intent, detect_city, TAG_SIM_BASE, TAG_SIM_SHORT

place_reviews = None
tfidf = None
tfidf_matrix = None
corpus_embeddings = None
tag_embeddings = None
tag_names = None
tag_norm = None

def load_search_backend(prepared_csv: str = "data/places_with_tags.csv", load_embeddings: bool = False, sbert_model_name: str = "all-MiniLM-L6-v2"):

    global place_reviews, tfidf, tfidf_matrix, corpus_embeddings, tag_embeddings, tag_names, tag_norm

    if not os.path.exists(prepared_csv):
        raise FileNotFoundError(f"Prepared CSV not found: {prepared_csv}. Run preparation first.")

    place_reviews = pd.read_csv(prepared_csv)
    if "full_text" not in place_reviews.columns:
        raise RuntimeError("prepared CSV missing 'full_text' column. Re-run preparation.")

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=50000)
    tfidf_matrix = tfidf.fit_transform(place_reviews["full_text"].astype(str).tolist())

    tag_names = list(TAG_QUERIES.keys())

    if load_embeddings:
        from src.tagging import init_sbert_model
        model = init_sbert_model(sbert_model_name)
        corpus_embeddings = model.encode(place_reviews["full_text"].astype(str).tolist(), batch_size=64, convert_to_tensor=True, show_progress_bar=True)
        tag_texts = [TAG_QUERIES[t] for t in tag_names]
        tag_embeddings = model.encode(tag_texts, convert_to_tensor=True)
        import torch.nn.functional as F
        corpus_embeddings = corpus_embeddings.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        tag_embeddings = tag_embeddings.to(corpus_embeddings.device)
        tag_norm = F.normalize(tag_embeddings, p=2, dim=1)

    globals().update({
        "place_reviews": place_reviews,
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "corpus_embeddings": corpus_embeddings,
        "tag_embeddings": tag_embeddings,
        "tag_names": tag_names,
        "tag_norm": tag_norm
    })

    print("Search backend loaded.")
    print(f"Places: {len(place_reviews)}. TF-IDF matrix shape: {tfidf_matrix.shape}")

def hybrid_search_with_tags(
    query,
    top_n=5,
    tag_filter=None,
    soft_boost=True,
    boost_weight=0.15,
    tag_detect_thresh=0.34,
    city_filter=None,
    rating_boost_weight=0.15
):
    global place_reviews, tfidf, tfidf_matrix, corpus_embeddings, tag_embeddings, tag_names, tag_norm

    if place_reviews is None or tfidf is None or tfidf_matrix is None:
        raise RuntimeError("Search backend not loaded. Call load_search_backend(prepared_csv, load_embeddings=...) first.")

    n_places = len(place_reviews)
    if n_places == 0:
        return place_reviews.iloc[0:0]

    q_vec = tfidf.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    if corpus_embeddings is not None:
        from sentence_transformers import util
        query_emb = corpus_embeddings.new_tensor([])
        try:
            sbert_scores = np.zeros(n_places, dtype=float)
        except Exception:
            sbert_scores = np.zeros(n_places, dtype=float)
    else:
        sbert_scores = np.zeros(n_places, dtype=float)

    final_scores = 0.4 * tfidf_scores + 0.6 * sbert_scores

    detected_city = None
    city_method = None
    city_mask = np.ones(n_places, dtype=bool)

    canonical_city_vocab = sorted([c for c in place_reviews.get("City_canon", pd.Series(dtype=str)).unique().tolist() if c and str(c).strip() != ""])
    try:
        cand_city, method, score = detect_city(query, canonical_city_vocab, alias_map=None)
        if cand_city:
            detected_city = cand_city
            city_method = method
            city_mask = (place_reviews["City_canon"].fillna("").astype(str) == detected_city).values
            if city_mask.sum() == 0:
                print(f"No places for '{query}' in '{detected_city}' in dataset. (strict city enforced)")
                return place_reviews.iloc[0:0][["City","Place","avg_rating","review_count","auto_tag"]].copy().assign(final_score=[])
    except Exception:
        pass

    mask = city_mask.copy()

    detected_tags = []
    detect_method = None
    from src.tagging import PHRASE_TO_TAG
    q_low = str(query).lower()
    for phrase, canonical_tag in PHRASE_TO_TAG.items():
        if phrase in q_low:
            detected_tags = [canonical_tag]
            detect_method = "phrase"
            break

    if not detected_tags and tag_embeddings is not None:
        try:
            from sentence_transformers import util
            tag_sim = np.zeros(len(tag_names))
            detected_idx = [i for i, s in enumerate(tag_sim) if s >= tag_detect_thresh]
            detected_tags = [tag_names[i] for i in detected_idx]
            detect_method = detect_method or "tag_sim"
        except Exception:
            detected_tags = []
            detect_method = None

    if tag_filter:
        if isinstance(tag_filter, str):
            tag_filter = [tag_filter]
        tag_mask = np.zeros(n_places, dtype=bool)
        for t in tag_filter:
            col = f"is_{t}"
            if col in place_reviews.columns:
                tag_mask = tag_mask | place_reviews[col].values
            else:
                print(f"Warning: tag '{t}' not found in place_reviews. Ignored.")
        combined_mask = mask & tag_mask
        if combined_mask.sum() == 0:
            if detected_city:
                print(f"No places matched tag_filter {tag_filter} in detected city '{detected_city}'. No results returned.")
            else:
                print(f"No places matched tag_filter {tag_filter}. No results returned.")
            return place_reviews.iloc[0:0][["City","Place","avg_rating","review_count","auto_tag"]].copy().assign(final_score=[])
        else:
            mask = combined_mask

    candidate_count_after_tag_filter = int(mask.sum())

    if soft_boost and detected_tags:
        for t in detected_tags:
            col = f"is_{t}"
            if col in place_reviews.columns:
                final_scores = final_scores + boost_weight * place_reviews[col].astype(float).values

    avg_ratings = place_reviews["avg_rating"].fillna(0).astype(float).values
    review_counts = place_reviews["review_count"].fillna(0).astype(float).values
    rating_confidence = avg_ratings * np.log1p(review_counts)
    if rating_confidence.max() > 0:
        rating_confidence_norm = rating_confidence / rating_confidence.max()
    else:
        rating_confidence_norm = np.zeros_like(rating_confidence)
    final_scores = final_scores + rating_boost_weight * rating_confidence_norm

    candidates_idx = np.where(mask)[0]
    if len(candidates_idx) == 0:
        print("No matching places after applying filters (city/tag).")
        return place_reviews.iloc[0:0][["City","Place","avg_rating","review_count","auto_tag"]].copy().assign(final_score=[])

    candidates_scores = final_scores[candidates_idx]
    order = np.argsort(candidates_scores)[::-1]
    top_rel_idx = candidates_idx[order[:top_n]]

    result = place_reviews.loc[top_rel_idx, ["City", "Place", "avg_rating", "review_count", "auto_tag"]].copy()
    result["final_score"] = final_scores[top_rel_idx]
    result = result.reset_index(drop=True)

    try:
        from src.tagging import print_diagnostics
        print_diagnostics(query, detected_city, city_method, detected_tags, int(city_mask.sum()), candidate_count_after_tag_filter)
    except Exception:
        pass

    return result

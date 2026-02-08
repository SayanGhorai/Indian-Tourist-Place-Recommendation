import os
import re
from pathlib import Path

import pandas as pd
import numpy as np

from typing import Optional

from src.tagging import TAG_QUERIES, PHRASE_TO_TAG, CITY_ALIAS_MAP

def _norm_city(s):
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def prepare_places(
    input_csv: str = "data/Review_db.csv",
    output_csv: str = "data/places_with_tags.csv",
    use_sbert: bool = False,
    sbert_model_name: str = "all-MiniLM-L6-v2",
    sbert_batch_size: int = 64,
    tag_sim_thresh: float = 0.30,
):

    input_csv = str(input_csv)
    output_csv = str(output_csv)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print("Loading:", input_csv)
    df = pd.read_csv(input_csv)

    df["Review"] = df["Review"].astype(str).str.strip()
    df = df[df["Review"].replace("", np.nan).notna()].copy()
    df["Rating"] = pd.to_numeric(df.get("Rating", pd.Series(dtype=float)), errors="coerce")
    df["Date"] = pd.to_datetime(df.get("Date", pd.Series()), errors="coerce")
    df["Name"] = df.get("Name").fillna("Anonymous")

    place_agg = df.groupby(["City", "Place"], dropna=False).agg(
        review_text = ("Review", lambda x: " ".join(x.astype(str))),
        avg_rating  = ("Rating", "mean"),
        review_count = ("Review", "count")
    ).reset_index()

    place_agg["full_text"] = (
        place_agg["City"].astype(str) + " " +
        place_agg["Place"].astype(str) + " " +
        place_agg["review_text"].astype(str)
    )
    place_agg["full_text"] = (
        place_agg["full_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    place_agg["City_raw"] = place_agg["City"].astype(str)
    place_agg["City_norm"] = place_agg["City_raw"].apply(_norm_city)

    def canonicalize_city(nc):
        if not nc:
            return ""
        if nc in CITY_ALIAS_MAP:
            return CITY_ALIAS_MAP[nc]
        return nc

    place_agg["City_canon"] = place_agg["City_norm"].apply(canonicalize_city)

    for tag in TAG_QUERIES.keys():
        place_agg[f"is_{tag}"] = False

    full_texts = place_agg["full_text"].astype(str).tolist()
    for phrase, tag in PHRASE_TO_TAG.items():
        if tag not in TAG_QUERIES:
            continue
        phrase_l = phrase.lower()
        mask = [phrase_l in ft for ft in full_texts]
        place_agg.loc[mask, f"is_{tag}"] = True

    for tag, tag_query in TAG_QUERIES.items():
        tokens = set([t.strip() for t in re.split(r"\W+", tag_query.lower()) if t.strip() and len(t) > 2])
        if not tokens:
            continue
        mask = np.zeros(len(place_agg), dtype=bool)
        for tok in tokens:
            mask = mask | place_agg["full_text"].str.contains(rf"\b{re.escape(tok)}\b", case=False, na=False)
        place_agg.loc[mask, f"is_{tag}"] = True

    if use_sbert:
        from src.tagging import init_sbert_model, compute_tag_embeddings
        model = init_sbert_model(sbert_model_name)
        place_agg = compute_tag_embeddings(model, place_agg, tag_sim_thresh=tag_sim_thresh, batch_size=sbert_batch_size)

    ensure_dir(os.path.dirname(output_csv) or ".")
    place_agg.to_csv(output_csv, index=False)
    print("Saved processed places CSV:", output_csv)
    return place_agg

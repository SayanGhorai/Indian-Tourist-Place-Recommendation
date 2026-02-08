import re
from difflib import get_close_matches
from typing import Tuple, List

TAG_QUERIES = {
    "hill": "hill station mountain viewpoint peak valley scenic hills ghats plateau",
    "beach": "beach sea ocean sand shoreline coast waves seaside sunset",
    "waterfall": "waterfall falls cascade water drop stream plunge pool",
    "lake": "lake reservoir boating paddle serene lakefront backwater",
    "river": "river ghat riverside rafting boating riverbank",
    "forest": "forest jungle greenery woods trees trekking trail wildlife park",
    "wildlife": "safari tiger elephant deer sanctuary national park animals birds",
    "desert": "desert dunes sand camel safari arid wasteland",
    "snow": "snow glacier skiing snowfall ice winter mountain",
    "temple": "mandir temple hindu shiva vishnu krishna hanuman devi aarti darshan prasad jyotirlinga",
    "mosque": "mosque masjid islam namaz salah quran minaret imam dargah",
    "church": "church cathedral chapel jesus mary christian cross bible mass",
    "gurdwara": "gurdwara sikh guru granth sahib langar khalsa gurudwara",
    "pilgrimage": "yatra pilgrimage devotees holy dip kumbh mela parikrama",
    "fort": "fort fortress rampart bastion cannon gate citadel defense walls",
    "palace": "palace royal king queen durbar mahal haveli",
    "monument": "statue memorial landmark tower arch structure landmark",
    "museum": "museum gallery exhibits artifacts history display collection",
    "ruins": "ruins archaeological remains broken walls excavation heritage site",
    "trekking": "trek hike hiking trail summit climb backpack route camp",
    "adventure": "rafting zipline paragliding camping climbing adventure sports",
    "shopping": "market bazaar shops souvenirs handicrafts street market vendors",
    "mall": "shopping mall brands cinema food court complex stores",
    "food": "street food dosa chaat lassi biryani thali sweets restaurant cafe eatery",
    "nightlife": "clubs bars pubs dj party nightlife drinks music dance",
    "sports": "stadium cricket football match ground sports arena",
    "garden": "garden park botanical flowers lawn picnic walking pathway",
    "island": "island ferry boat coral beach isolated lagoon snorkeling",
    "romantic": "romantic couples honeymoon sunset candlelight scenic viewpoint",
    "family_friendly": "kids playground zoo amusement park family picnic boating",
    "photography": "photogenic viewpoint skyline landscape sunset sunrise instagram",
    "wellness": "spa ayurveda yoga meditation massage therapy retreat",
    "festival": "festival mela fair celebration procession cultural event dance music",
    "safety": "safe clean maintained secure lighting police family safe area"
}

PHRASE_TO_TAG = {
    "street food": "food",
    "local food": "food",
    "best dosa": "food",
    "hill station": "hill",
    "mountain view": "hill",
    "waterfall": "waterfall",
    "historic fort": "fort",
    "old fort": "fort",
    "shopping mall": "shopping",
    "family activities": "family_friendly",
}

CITY_ALIAS_MAP = {
    "bangalore": "bengaluru",
    "bangaluru": "bengaluru",
    "bengaluru": "bengaluru",
    "mumabi": "mumbai",
    "mumbai": "mumbai",
    "new delhi": "delhi",
    "delhi": "delhi",
    "gurugram": "gurgaon",
    "gurgaon": "gurgaon",
    "pondicherry": "puducherry",
    "puducherry": "puducherry",
}

def init_sbert_model(model_name: str = "all-MiniLM-L6-v2", device: str = None):

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers") from e

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model

def compute_tag_embeddings(model, place_reviews, tag_sim_thresh: float = 0.30, batch_size: int = 64):

    import torch
    import torch.nn.functional as F
    from sentence_transformers import util
    import numpy as np

    texts = place_reviews["full_text"].astype(str).tolist()
    corpus_embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    tag_texts = [TAG_QUERIES[k] for k in TAG_QUERIES.keys()]
    tag_embeddings = model.encode(tag_texts, convert_to_tensor=True)

    corpus_norm = F.normalize(corpus_embeddings, p=2, dim=1)
    tag_norm = F.normalize(tag_embeddings, p=2, dim=1)
    scores = util.cos_sim(corpus_norm, tag_norm).cpu().numpy()  # (n_places, n_tags)

    top_idx = scores.argmax(axis=1)
    place_reviews = place_reviews.copy()
    tag_names = list(TAG_QUERIES.keys())
    place_reviews["auto_tag"] = [tag_names[i] for i in top_idx]
    place_reviews["auto_tag_score"] = scores[np.arange(scores.shape[0]), top_idx].tolist()
    place_reviews["auto_tags"] = [[tag_names[i] for i in row.argsort()[::-1][:3]] for row in scores]

    for i, tag in enumerate(tag_names):
        place_reviews[f"is_{tag}"] = scores[:, i] >= tag_sim_thresh

    place_reviews.attrs["corpus_embeddings"] = corpus_embeddings.cpu().numpy()
    place_reviews.attrs["tag_embeddings"] = tag_embeddings.cpu().numpy()
    return place_reviews

CITY_FUZZY_THRESH = 0.75

def detect_city(query: str, canonical_vocab: list, alias_map: dict = CITY_ALIAS_MAP, fuzzy_cutoff: float = CITY_FUZZY_THRESH) -> Tuple[object, str, float]:
    """
    Returns (detected_city_canon_or_None, method, score)
    method in {"direct", "alias", "fuzzy", None}
    """
    q = str(query).lower()
    q = re.sub(r"[^\w\s]", " ", q)
    tokens = q.split()
    for tok in tokens[::-1]:
        tok_norm = tok.strip()
        if not tok_norm:
            continue
        if tok_norm in canonical_vocab:
            return tok_norm, "direct", 1.0
        if tok_norm in alias_map:
            return alias_map[tok_norm], "alias", 1.0
    words = [t for t in tokens if t]
    n = len(words)
    for L in (3, 2):
        if n < L:
            continue
        for i in range(n - L + 1):
            cand = " ".join(words[i:i+L])
            cand_norm = cand.strip()
            if cand_norm in canonical_vocab:
                return cand_norm, "direct", 1.0
            if cand_norm in alias_map:
                return alias_map[cand_norm], "alias", 1.0
    candidate_pool = list(set(tokens + [" ".join(tokens[i:j]) for i in range(len(tokens)) for j in range(i+1, min(i+4, len(tokens))+1)]))
    candidate_pool = [c for c in candidate_pool if c]
    for cand in candidate_pool:
        matches = get_close_matches(cand, canonical_vocab, n=1, cutoff=fuzzy_cutoff)
        if matches:
            return matches[0], "fuzzy", fuzzy_cutoff
    return None, None, 0.0

TAG_SIM_BASE = 0.65
TAG_SIM_SHORT = 0.55

def detect_intent(query: str, model, tag_norm, tag_names: list, base_thresh: float = TAG_SIM_BASE, short_thresh: float = TAG_SIM_SHORT):
    """
    Returns (candidates_list, method)
    method in {"phrase", "sbert"}
    """
    q = str(query).lower()
    for phrase, canonical_tag in PHRASE_TO_TAG.items():
        if phrase in q and canonical_tag in tag_names:
            return [canonical_tag], "phrase"
    from sentence_transformers import util
    import torch
    q_emb = model.encode(q, convert_to_tensor=True, device=tag_norm.device)
    q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=0)
    sims = util.cos_sim(q_emb, tag_norm).cpu().numpy().ravel()
    token_count = len(q.split())
    thresh = short_thresh if token_count <= 3 else base_thresh
    candidate_idx = [i for i, s in enumerate(sims) if s >= thresh]
    candidate_idx = sorted(candidate_idx, key=lambda i: -sims[i])
    candidates = [tag_names[i] for i in candidate_idx]
    return candidates, "sbert"

def print_diagnostics(query, detected_city, city_method, detected_tags, city_mask_sum, candidate_count_after_tag_filter, extra=None):
    print("=== DIAGNOSTICS ===")
    print(f"Query: {query}")
    print(f"Detected city: {detected_city} (method={city_method})")
    print(f"Detected tags: {detected_tags}")
    print(f"city_mask.sum(): {city_mask_sum}")
    print(f"candidates after tag filter: {candidate_count_after_tag_filter}")
    if extra:
        print("Extra:", extra)
    print("===================\n")

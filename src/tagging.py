import re
from difflib import get_close_matches
from typing import Tuple

# ---------------- TAG QUERIES ----------------
TAG_QUERIES = {
    "hill": "hill station mountain viewpoint scenic hills valley",
    "beach": "beach sea ocean sand sunset coast waves",
    "waterfall": "waterfall cascade stream plunge pool",
    "lake": "lake boating reservoir serene backwater",
    "river": "river ghat rafting boating riverside",
    "forest": "forest jungle greenery trekking wildlife",
    "wildlife": "safari tiger elephant sanctuary birds",
    "desert": "desert dunes camel safari sand",
    "snow": "snow glacier ice winter mountain",

    "temple": "temple mandir hindu darshan jyotirlinga",
    "mosque": "mosque masjid islam namaz dargah",
    "church": "church cathedral chapel christian mass",
    "gurdwara": "gurdwara sikh langar khalsa",
    "pilgrimage": "pilgrimage yatra holy devotees",
    "spiritual": "spiritual holy sacred divine meditation blessings",

    "fort": "fort fortress bastion defense walls",
    "palace": "palace royal king queen mahal haveli",
    "monument": "monument memorial tower arch landmark",
    "museum": "museum artifacts history gallery",
    "ruins": "ruins archaeological remains heritage",

    "trekking": "trek hike summit trail camp",
    "adventure": "rafting zipline paragliding camping",
    "shopping": "market bazaar handicrafts souvenirs",
    "food": "street food dosa biryani sweets cafe restaurant",
    "street_food": "chaat dosa biryani snacks stalls vendors",
    "fine_dining": "fine dining luxury restaurant premium cuisine",
    "nightlife": "clubs bars pubs dj dance music",

    "romantic": "honeymoon couples sunset romantic",
    "family_friendly": "kids family picnic zoo boating",
    "photography": "photogenic viewpoint skyline sunrise sunset",
    "wellness": "spa ayurveda yoga meditation retreat",

    "peaceful": "peaceful calm serene quiet relaxing less crowded",
    "crowded": "crowded rush packed busy tourists",
    "budget": "cheap affordable budget low cost",
    "luxury": "luxury premium expensive resort high-end",
    "solo_travel": "solo backpacker independent traveler alone",
    "road_trip": "road trip scenic drive highway long drive",
    "local_experience": "local culture village tradition authentic heritage"
}

# ---------------- PHRASE MAPPING ----------------
PHRASE_TO_TAG = {
    "street food": "street_food",
    "local food": "food",
    "peaceful place": "peaceful",
    "less crowded": "peaceful",
    "family trip": "family_friendly",
    "honeymoon trip": "romantic",
    "mountain view": "hill",
    "historic fort": "fort",
    "old fort": "fort",
    "budget trip": "budget",
    "luxury stay": "luxury",
    "local culture": "local_experience",
    "road trip": "road_trip",
    "solo trip": "solo_travel",
    "holy temple": "spiritual"
}

# ---------------- CITY ALIAS ----------------
CITY_ALIAS_MAP = {
    "bangalore": "bengaluru",
    "bangaluru": "bengaluru",
    "bombay": "mumbai",
    "mumabi": "mumbai",
    "new delhi": "delhi",
    "ncr": "delhi",
    "calcutta": "kolkata",
    "madras": "chennai",
    "pondicherry": "puducherry",

    # Goa mapping
    "goa": "goa",
    "north goa": "goa",
    "south goa": "goa",
    "goa velha": "goa",
    "old goa": "goa",
    "panjim": "goa",
    "panaji": "goa",
    "calangute": "goa",
    "baga": "goa",
    "candolim": "goa",
    "arambol": "goa"
}

# ---------------- SBERT ----------------
def init_sbert_model(model_name="all-MiniLM-L6-v2", device=None):
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(model_name, device=device)


# ---------------- CITY DETECTION ----------------
CITY_FUZZY_THRESH = 0.75

def detect_city(query, canonical_vocab, alias_map=CITY_ALIAS_MAP, fuzzy_cutoff=CITY_FUZZY_THRESH):

    q = str(query).lower()
    q = re.sub(r"[^\w\s]", " ", q)
    tokens = q.split()

    for tok in tokens[::-1]:
        if tok in canonical_vocab:
            return tok, "direct", 1.0
        if tok in alias_map:
            return alias_map[tok], "alias", 1.0

    for cand in tokens:
        matches = get_close_matches(cand, canonical_vocab, n=1, cutoff=fuzzy_cutoff)
        if matches:
            return matches[0], "fuzzy", fuzzy_cutoff

    return None, None, 0.0


# ---------------- INTENT DETECTION ----------------
TAG_SIM_BASE = 0.50
TAG_SIM_SHORT = 0.40

def detect_intent(query, model, tag_norm, tag_names):

    q = str(query).lower()

    for phrase, canonical_tag in PHRASE_TO_TAG.items():
        if phrase in q and canonical_tag in tag_names:
            return [canonical_tag], "phrase"

    from sentence_transformers import util
    import torch

    q_emb = model.encode(q, convert_to_tensor=True)
    q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=0)

    sims = util.cos_sim(q_emb, tag_norm).cpu().numpy().ravel()

    token_count = len(q.split())
    thresh = TAG_SIM_SHORT if token_count <= 3 else TAG_SIM_BASE

    candidate_idx = [i for i, s in enumerate(sims) if s >= thresh]
    candidate_idx = sorted(candidate_idx, key=lambda i: -sims[i])

    candidates = [tag_names[i] for i in candidate_idx]

    return candidates, "sbert"


# ---------------- DIAGNOSTICS ----------------
def print_diagnostics(query, detected_city, city_method, detected_tags, city_mask_sum, candidate_count_after_tag_filter):
    print("=== DIAGNOSTICS ===")
    print(f"Query: {query}")
    print(f"Detected city: {detected_city}")
    print(f"Method: {city_method}")
    print(f"Detected tags: {detected_tags}")
    print(f"Candidates: {candidate_count_after_tag_filter}")
    print("===================")
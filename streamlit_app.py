import streamlit as st
import pandas as pd

from src.search_engine import (
    load_search_backend,
    hybrid_search_with_tags
)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Hybrid GenAI Tourist Recommender",
    page_icon="🌍",
    layout="wide"
)

# ---------------- Load Backend ----------------
@st.cache_resource
def load_backend():
    load_search_backend(
        prepared_csv="data/places_genai_ready.csv",
        load_embeddings=True
    )

load_backend()

# ---------------- Header ----------------
st.title("🌍 Hybrid GenAI Tourist Place Recommendation System")
st.markdown(
    """
    Discover the best tourist destinations across India using **Hybrid AI Retrieval (TF-IDF + SBERT)**  
    with **semantic intent understanding, explainable ranking, and personalized recommendations**.
    """
)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙ Search Settings")

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# ---------------- Search Input ----------------
query = st.text_input(
    "🔎 Enter your travel preference",
    placeholder="Example: peaceful beach in goa"
)

search_btn = st.button("Find Places")

# ---------------- Search ----------------
if search_btn:

    if not query.strip():
        st.warning("Please enter a travel query.")
    else:
        with st.spinner("Finding best places for you..."):
            results = hybrid_search_with_tags(
                query=query,
                top_n=top_n
            )

        if results.empty:
            st.error("No places found for this query.")
        else:
            st.success(f"Found {len(results)} recommendations")

            for idx, row in results.iterrows():

                st.markdown("---")

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader(f"📍 {row['Place']}")
                    st.caption(f"City: {row['City']}")

                with col2:
                    st.metric(
                        label="⭐ Rating",
                        value=round(float(row["avg_rating"]), 2)
                    )

                # Tags
                st.markdown("### 🏷 Tags")
                if isinstance(row["auto_tags"], str):
                    st.write(row["auto_tags"])
                else:
                    st.write(", ".join(row["auto_tags"]))

                # Keywords
                st.markdown("### 🔑 Top Keywords")
                if "top_keywords" in row and pd.notna(row["top_keywords"]):
                    st.write(row["top_keywords"])

                # Pros / Cons
                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("### 👍 Pros")
                    if pd.notna(row["pros"]):
                        st.success(row["pros"])
                    else:
                        st.write("No major positives extracted.")

                with col4:
                    st.markdown("### ⚠ Cons")
                    if pd.notna(row["cons"]):
                        st.warning(row["cons"])
                    else:
                        st.write("No major negatives extracted.")

                # AI Explanation
                st.markdown("### 🧠 Why Recommended?")
                if pd.notna(row["recommendation_explanation"]):
                    st.info(row["recommendation_explanation"])

                # Score
                st.progress(min(float(row["final_score"]), 1.0))

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    """
    Built with:
    - **TF-IDF**
    - **Sentence-BERT**
    - **Hybrid Retrieval**
    - **Intent Detection**
    - **Explainable AI**
    """
)
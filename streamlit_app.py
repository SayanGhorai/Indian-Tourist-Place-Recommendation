import streamlit as st
import pandas as pd

from src.search_engine import (
    load_search_backend,
    hybrid_search_with_tags
)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="GenAI Indian Tourist Place Recommendation System",
    page_icon="🇮🇳",
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

# ---------------- Sidebar ----------------
st.sidebar.title("ℹ About Project")

st.sidebar.markdown("""
### 📌 This application uses:

- **Semantic Search (TF-IDF + SBERT)**
- **Intent Detection**
- **City-aware Filtering**
- **Explainable Recommendations**
- **Pros / Cons Extraction**

---

### 📂 Dataset
Kaggle Indian Places to Visit Reviews Dataset

---

### ⚙ Built With

- Python
- Pandas
- Scikit-learn
- Sentence Transformers
- Streamlit
""")

top_n = st.sidebar.slider(
    "Recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# ---------------- Main Header ----------------
st.title("🇮🇳 GenAI Indian Tourist Place Recommendation System")

st.markdown("""
Explore India's most loved destinations with an AI-powered travel recommendation engine.  
Get personalized suggestions based on your interests, travel style, and intent — powered by semantic understanding, intelligent ranking, and explainable insights from real traveler reviews.
""")

# ---------------- Search ----------------
query = st.text_input(
    "🔍 What kind of place are you looking for?",
    placeholder="Example: peaceful beach in goa"
)

search_btn = st.button("Find Places")

# ---------------- Results ----------------
if search_btn:

    if not query.strip():
        st.warning("Please enter your travel preference.")

    else:
        with st.spinner("Finding best places for you..."):

            results = hybrid_search_with_tags(
                query=query,
                top_n=top_n
            )

        if results.empty:
            st.error("No places found.")

        else:
            st.success(f"Top {len(results)} recommendations for: {query}")

            for idx, row in results.iterrows():

                with st.container():

                    st.markdown("---")

                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.subheader(f"{idx+1}. 📍 {row['Place']}")
                        st.caption(f"📌 {row['City']}")

                    with col2:
                        st.metric(
                            "⭐ Rating",
                            round(float(row["avg_rating"]), 2)
                        )

                    # Compact AI Summary
                    st.markdown("### 🧠 Why this place?")

                    if pd.notna(row["recommendation_explanation"]):
                        st.info(row["recommendation_explanation"])

                    # Small compact tags
                    if isinstance(row["auto_tags"], list):
                        st.caption(
                            "🏷 " + " | ".join(row["auto_tags"][:3])
                        )

                    # Expandable details
                    with st.expander("View Details"):

                        if isinstance(row["top_keywords"], list):
                            st.markdown(
                                f"**Keywords:** {', '.join(row['top_keywords'][:5])}"
                            )

                        col3, col4 = st.columns(2)

                        with col3:
                            if isinstance(row["pros"], list) and len(row["pros"]) > 0:
                                st.success(
                                    "Pros: " + ", ".join(row["pros"][:5])
                                )

                        with col4:
                            if isinstance(row["cons"], list) and len(row["cons"]) > 0:
                                st.warning(
                                    "Cons: " + ", ".join(row["cons"][:5])
                                )

                        st.progress(
                            min(float(row["final_score"]), 1.0)
                        )

                        st.caption(
                            f"Confidence Score: {round(float(row['final_score']), 3)}"
                        )
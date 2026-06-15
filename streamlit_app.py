import streamlit as st
import pandas as pd

from src.search_engine import (
    load_search_backend,
    hybrid_search_with_tags
)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="GenAI Indian Tourist Place Recommendation System",
    page_icon="🗺️",
    layout="wide"
)

# ---------------- Load Backend ----------------
@st.cache_resource
def load_backend():
    try:
        load_search_backend(
            prepared_csv="data/places_genai_ready.csv",
            load_embeddings=True
        )
        return True
    except Exception as e:
        return str(e)

backend_loaded = load_backend()

# Show backend error if failed
if backend_loaded != True:
    st.error(f"Backend loading failed: {backend_loaded}")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 📌 About")

    st.info("""
**GenAI Travel Recommendation**

• TF-IDF + SBERT  
• Intent Detection  
• City Filtering  
• Explainable AI  
• Semantic Ranking  
""")

    st.markdown("### 📂 Dataset")
    st.caption("""
Kaggle Indian Places Dataset  
1.5M Reviews  
1782 Cleaned Cities
""")

    st.markdown("### ⚙ Built With")
    st.caption("""
Python • Pandas • Scikit-learn  
Sentence Transformers • Streamlit
""")

    top_n = st.slider(
        "Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

# ---------------- Main ----------------
st.title("GenAI Indian Tourist Place Recommendation System")

st.caption(
    "Get recommendations for the best places in India using semantic AI search and explainable recommendations."
)

query = st.text_input(
    "Search your travel preference",
    placeholder="Example: best street food in Delhi"
)

search_btn = st.button("Get Recommendations")

# ---------------- Results ----------------
if search_btn:

    if not query.strip():
        st.warning("Please enter a query.")

    else:
        with st.spinner("Searching..."):

            results = hybrid_search_with_tags(
                query=query,
                top_n=top_n
            )

        if results.empty:
            st.error("No places found.")

        else:
            st.success(f"Showing top {len(results)} places")

            # Table View
            table_df = results[
                [
                    "City",
                    "Place",
                    "avg_rating",
                    "auto_tags",
                    "top_keywords",
                    "final_score"
                ]
            ].copy()

            st.dataframe(
                table_df,
                use_container_width=True
            )

            st.markdown("## Place Highlights")

            for idx, row in results.iterrows():

                with st.expander(
                    f"{idx+1}. {row['Place']} ({row['City']})"
                ):

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write("### Why recommended?")
                        st.info(
                            row["recommendation_explanation"]
                        )

                    with col2:
                        st.metric(
                            "⭐ Rating",
                            round(float(row["avg_rating"]), 2)
                        )

                    st.write("**Tags:**")
                    st.write(
                        ", ".join(row["auto_tags"][:5])
                    )

                    st.write("**Top Keywords:**")
                    st.write(
                        ", ".join(row["top_keywords"][:5])
                    )

                    col3, col4 = st.columns(2)

                    with col3:
                        st.success(
                            "Pros: " + ", ".join(row["pros"][:5])
                        )

                    with col4:
                        st.warning(
                            "Cons: " + ", ".join(row["cons"][:5])
                        )

                    st.progress(
                        min(float(row["final_score"]), 1.0)
                    )

                    st.caption(
                        f"Confidence Score: {round(float(row['final_score']), 3)}"
                    )
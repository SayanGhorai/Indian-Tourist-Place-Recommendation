import pandas as pd

df = pd.read_csv("data/places_with_tags.csv")

print("Rows:", len(df))
print("\nColumns:\n", df.columns.tolist())
print("\nSample:\n")
print(df[["City","Place","avg_rating","review_count","auto_tag","auto_tag_score"]].head(10))

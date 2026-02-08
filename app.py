import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/Review_db.csv", help="Path to raw Review_db.csv")
    parser.add_argument("--prepared-csv", default="data/places_with_tags.csv", help="Path to prepared place-level CSV")
    parser.add_argument("--prepare", action="store_true", help="Run data preparation (creates prepared CSV)")
    parser.add_argument("--use-sbert", action="store_true", help="Compute SBERT-based tags when preparing (slow)")
    parser.add_argument("--load-embeddings", action="store_true", help="Load / compute SBERT embeddings for search (slow)")
    args = parser.parse_args()

    if args.prepare:
        from src.data_loader import prepare_places
        prepare_places(input_csv=args.input_path, output_csv=args.prepared_csv, use_sbert=args.use_sbert)
        print("Preparation complete.")
        return

    if not Path(args.prepared_csv).exists():
        print(f"Prepared CSV not found at {args.prepared_csv}. Run with --prepare first.")
        return

    from src.search_engine import load_search_backend, hybrid_search_with_tags
    load_search_backend(prepared_csv=args.prepared_csv, load_embeddings=args.load_embeddings)

    print("Interactive search ready. Type 'exit' to quit.")
    while True:
        q = input("Enter query: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        res = hybrid_search_with_tags(q, top_n=5, city_filter=None)
        if res is None or res.shape[0] == 0:
            print("No results.")
        else:
            print(res.to_string(index=False))
    print("Goodbye.")

if __name__ == "__main__":
    main()

from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepared-csv",
        default="data/places_genai_ready.csv",
        help="Path to final prepared CSV"
    )

    parser.add_argument(
        "--load-embeddings",
        action="store_true",
        help="Load SBERT embeddings for semantic search"
    )

    args = parser.parse_args()

    # Check CSV exists
    if not Path(args.prepared_csv).exists():
        print(f"File not found: {args.prepared_csv}")
        return

    # Load backend
    from src.search_engine import (
        load_search_backend,
        hybrid_search_with_tags
    )

    load_search_backend(
        prepared_csv=args.prepared_csv,
        load_embeddings=args.load_embeddings
    )

    print("\nHybrid GenAI Tourist Recommendation System Ready")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter query: ").strip()

        if not query or query.lower() in {"exit", "quit"}:
            break

        results = hybrid_search_with_tags(
            query=query,
            top_n=5
        )

        if results.empty:
            print("\nNo results found.\n")
        else:
            print("\nTop Recommendations:\n")
            print(results.to_string(index=False))
            print("\n")

    print("Goodbye.")


if __name__ == "__main__":
    main()
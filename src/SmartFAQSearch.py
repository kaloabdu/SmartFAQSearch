
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "faqs.csv"
DB_PATH = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "university_faqs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_faqs():
    """Load and clean FAQ data from the CSV file."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"FAQ file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_columns = {"question", "answer"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain question and answer columns.")

    df = df.dropna(subset=["question", "answer"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    df = df[
        (df["question"] != "") &
        (df["answer"] != "")
    ].reset_index(drop=True)

    if "category" not in df.columns:
        df["category"] = "General"
    else:
        df["category"] = df["category"].fillna("General").astype(str)

    if df.empty:
        raise ValueError("No valid FAQs were found in the CSV file.")

    return df


def build_search_index(df, model):
    """Convert FAQ questions into embeddings and store them."""
    client = chromadb.PersistentClient(path=str(DB_PATH))

    existing_names = [
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    ]

    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    questions = df["question"].tolist()
    embeddings = model.encode(questions).tolist()

    collection.add(
        ids=[f"faq-{i}" for i in range(len(df))],
        documents=questions,
        embeddings=embeddings,
        metadatas=[
            {
                "answer": row["answer"],
                "category": row["category"],
            }
            for _, row in df.iterrows()
        ],
    )

    return collection




def search_faqs(query, collection, model, top_k=3):
    """Find relevant FAQs and ask for clarification when matches are ambiguous."""
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        print("Sorry, no FAQs are available to search.")
        return

    questions = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    DISTANCE_THRESHOLD = 0.65
    AMBIGUITY_MARGIN = 0.08

    best_distance = distances[0]

    if best_distance > DISTANCE_THRESHOLD:
        print(
            "\nSorry, I couldn't find a sufficiently relevant FAQ "
            "for your question. Please try rephrasing it or contact "
            "university support."
        )
        print(f"Closest match distance: {best_distance:.4f}")
        return

    # Keep only matches that pass the relevance threshold.
    candidates = [
        (question, metadata, distance)
        for question, metadata, distance in zip(
            questions, metadatas, distances
        )
        if distance <= DISTANCE_THRESHOLD
    ]

    # If the first two matches are close, the user's intent may be ambiguous.
    is_ambiguous = (
        len(candidates) >= 2
        and candidates[1][2] - candidates[0][2] <= AMBIGUITY_MARGIN
    )

    if is_ambiguous:
        question, metadata, distance = candidates[0]

        print("\nYour question may have more than one interpretation.")
        print("\nClosest possible answer:")
        print(f"Question: {question}")
        print(f"Answer: {metadata['answer']}")
        print(f"Category: {metadata['category']}")
        print(f"Distance: {distance:.4f}")

        print("\nDid you mean one of these questions?")
        for index, (candidate_question, _, candidate_distance) in enumerate(
            candidates, start=1
        ):
            print(
                f"{index}. {candidate_question} "
                f"(distance: {candidate_distance:.4f})"
            )

        print(
            "\nIf the closest answer matches what you meant, "
            "you can use it. Otherwise, choose the question "
            "that best matches your intention."
        )
        return

    # For a clear match, display only the best answer.
    question, metadata, distance = candidates[0]

    print("\nBest match")
    print(f"Question: {question}")
    print(f"Answer: {metadata['answer']}")
    print(f"Category: {metadata['category']}")
    print(f"Distance: {distance:.4f}")

def main():
    print("Loading FAQ data...")
    faqs = load_faqs()
    print(f"Loaded {len(faqs)} FAQs.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building search index...")
    collection = build_search_index(faqs, model)
    print("SmartFAQSearch is ready!")

    while True:
        query = input(
            "\nAsk a question (or type 'exit' to quit): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        search_faqs(query, collection, model)


if __name__ == "__main__":
    main()

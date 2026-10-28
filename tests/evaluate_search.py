
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from src.SmartFAQSearch import load_faqs, build_search_index, MODEL_NAME


# Keep this fixed while collecting evaluation results.
DISTANCE_THRESHOLD = 0.65

# Each expected question must match the CSV wording exactly.
RELEVANT_TESTS = [
    (
        "I forgot my login password. What should I do?",
        "How can I reset my password?",
    ),
    (
        "I can't log into my account. How can I get back in?",
        "How can I reset my password?",
    ),
    (
        "I need help with my university account.",
        "How can I get help with a student account?",
    ),
    (
        "Who can help me with a student account problem?",
        "How can I get help with a student account?",
    ),
    (
        "Where can I find the materials for my classes?",
        "How do I access course materials?",
    ),
    (
        "How do I open the learning materials my instructor shared?",
        "How do I access course materials?",
    ),
    (
        "Where can I check my marks?",
        "Where can I find my grades?",
    ),
    (
        "How do I see my exam results?",
        "Where can I find my grades?",
    ),
    (
        "How can I get in touch with my lecturer?",
        "How do I contact my instructor?",
    ),
    (
        "Where can I find my instructor's contact details?",
        "How do I contact my instructor?",
    ),
    (
        "When and where can I see my examination timetable?",
        "Where can I find my exam schedule?",
    ),
    (
        "How can I check the dates of my exams?",
        "Where can I find my exam schedule?",
    ),
    (
        "I want to change the courses I registered for.",
        "How do I change my course registration?",
    ),
    (
        "Who should I contact to modify my course registration?",
        "How do I change my course registration?",
    ),
]

# These questions are outside the information covered by our FAQs.
OUT_OF_SCOPE_TESTS = [
    "What is the best restaurant in Addis Ababa?",
    "How do I repair a car engine?",
    "Who won the World Cup?",
    "What is the capital of Japan?",
    "Can you write a Python program to sort a list?",
    "How can I recover a permanently deleted computer file?",
    "What is the weather forecast for tomorrow?",
    "Can you explain how a combustion engine works?",
]


def get_top_result(query, collection, model):
    """Return the closest FAQ and its cosine distance."""
    result = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=1,
        include=["documents", "distances"],
    )

    if not result["documents"] or not result["documents"][0]:
        return None, float("inf")

    return (
        result["documents"][0][0],
        result["distances"][0][0],
    )


def main():
    print("Loading FAQ data...")
    faqs = load_faqs()
    available_questions = set(faqs["question"])

    # Check test data before running the model.
    missing = sorted({
        expected
        for _, expected in RELEVANT_TESTS
        if expected not in available_questions
    })

    if missing:
        print("\nERROR: These expected questions were not found in data/faqs.csv:")
        for question in missing:
            print(f"- {question}")
        print("\nCorrect the wording in RELEVANT_TESTS and run again.")
        return

    print(f"Loaded {len(faqs)} FAQs.")
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building search index...")
    collection = build_search_index(faqs, model)

    
    # Diagnose the exam-results query by inspecting its top 3 matches.
    diagnostic_query = "How do I see my exam results?"

    diagnostic = collection.query(
        query_embeddings=model.encode([diagnostic_query]).tolist(),
        n_results=min(3, collection.count()),
        include=["documents", "distances"],
    )

    print("\n=== EXAM RESULTS DIAGNOSTIC ===")
    print(f"Query: {diagnostic_query}")

    for rank, (question, distance) in enumerate(
        zip(
            diagnostic["documents"][0],
            diagnostic["distances"][0],
        ),
        start=1,
    ):
        print(f"{rank}. Distance: {distance:.4f} | {question}")

    # Evaluate relevant queries.
    correct = 0
    wrongly_rejected = 0

    print("\n=== RELEVANT QUERY EVALUATION ===")

    for query, expected in RELEVANT_TESTS:
        predicted, distance = get_top_result(query, collection, model)

        accepted = distance <= DISTANCE_THRESHOLD
        matched = predicted == expected

        if matched and accepted:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
            if not accepted:
                wrongly_rejected += 1

        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Distance: {distance:.4f}")
        print(f"Accepted: {accepted}")
        print(f"Result: {status}")

    # Evaluate questions outside the FAQ scope.
    rejected = 0
    false_acceptances = 0

    print("\n=== OUT-OF-SCOPE QUERY EVALUATION ===")

    for query in OUT_OF_SCOPE_TESTS:
        predicted, distance = get_top_result(query, collection, model)
        was_rejected = distance > DISTANCE_THRESHOLD

        if was_rejected:
            rejected += 1
            status = "PASS"
        else:
            false_acceptances += 1
            status = "FALSE ACCEPTANCE"

        print(f"\nQuery: {query}")
        print(f"Closest FAQ: {predicted}")
        print(f"Distance: {distance:.4f}")
        print(f"Decision: {'REJECT' if was_rejected else 'ACCEPT'}")
        print(f"Result: {status}")

    # Summary metrics.
    relevant_total = len(RELEVANT_TESTS)
    out_of_scope_total = len(OUT_OF_SCOPE_TESTS)

    retrieval_accuracy = correct / relevant_total * 100
    rejection_rate = rejected / out_of_scope_total * 100

    print("\n" + "=" * 42)
    print("EVALUATION SUMMARY")
    print("=" * 42)
    print(f"Distance threshold: {DISTANCE_THRESHOLD:.2f}")
    print(
        f"Relevant queries retrieved correctly: "
        f"{correct}/{relevant_total} ({retrieval_accuracy:.1f}%)"
    )
    print(
        f"Relevant queries wrongly rejected: "
        f"{wrongly_rejected}/{relevant_total}"
    )
    print(
        f"Out-of-scope queries rejected: "
        f"{rejected}/{out_of_scope_total} ({rejection_rate:.1f}%)"
    )
    print(f"False acceptances: {false_acceptances}/{out_of_scope_total}")

    print(
        "\nThese results measure only this small test set. "
        "Review failures before changing the threshold."
    )


if __name__ == "__main__":
    main()

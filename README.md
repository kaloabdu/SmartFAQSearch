# SmartFAQSearch 🔎

A beginner-friendly semantic search application that helps users find relevant answers from university FAQs using natural-language questions.

SmartFAQSearch uses **Sentence Transformers** to convert FAQ questions into vector embeddings and **ChromaDB** to retrieve semantically similar questions. It can handle different ways of asking a question, reject queries that appear unrelated to the FAQ dataset, and identify potentially ambiguous results.

## ✨ Features

- **Semantic search:** Find relevant FAQs even when a user's wording differs from the original question.
- **FAQ data processing:** Load, validate, and clean FAQ data from a CSV file using Pandas.
- **Vector embeddings:** Represent FAQ questions as numerical embeddings using Sentence Transformers.
- **Vector database:** Store and retrieve embeddings using ChromaDB.
- **Relevance filtering:** Reject queries whose nearest match falls outside a configured distance threshold.
- **Ambiguity handling:** Present alternatives when multiple matches are sufficiently close.
- **Testing and evaluation:** Include unit tests and a separate evaluation script that uses the real embedding model.

## 🛠️ Technologies

| Technology                  | Purpose                                 |
| --------------------------- | --------------------------------------- |
| Python 3.12                 | Application development                 |
| Pandas 3.1.0                | Loading and cleaning FAQ data           |
| Sentence Transformers 6.2.0 | Generating text embeddings              |
| ChromaDB 1.5.9              | Storing and searching vector embeddings |
| `unittest`                  | Testing search behavior                 |

## ⚙️ How It Works

1. **Load the FAQ dataset:** Read questions, answers, and optional categories from `data/faqs.csv`.
2. **Generate embeddings:** Use a Sentence Transformers model to convert FAQ questions into numerical vectors.
3. **Build the search index:** Store the question embeddings and associated answers in ChromaDB.
4. **Process the user's query:** Convert the user's question into an embedding and retrieve the closest FAQ matches.
5. **Evaluate relevance:** Compare the closest match's distance against a configured threshold.
6. **Respond:** Return the best matching answer, present alternatives for potentially ambiguous queries, or reject a query that appears out of scope.

Semantic search compares text embeddings rather than relying only on exact keyword matches. However, semantic similarity does not guarantee that the retrieved answer is correct.

## 📁 Project Structure

```text
week-02/
├── data/
│   └── faqs.csv
├── src/
│   └── SmartFAQSearch.py
├── tests/
│   ├── test_search.py
│   └── evaluate_search.py
├── .gitignore
├── README.md
└── requirements.txt
```

The `.venv/` virtual environment, generated `chroma_db/` data, and Python cache files are local development artifacts and should not be committed to GitHub.

## 🚀 Getting Started

### Requirements

- Python 3.12
- Internet access to install dependencies and download the embedding model on its first run

The commands below are for Windows PowerShell.

### 1. Open the project directory

Navigate to the `week-02` project root:

```powershell
cd "G:\Projects\AWS projects\week-02"
```

### 2. Create a virtual environment

If you do not already have one:

```powershell
py -3.12 -m venv .venv
```

If your existing `.venv` works, keep using it instead of recreating it.

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Check dependencies

```powershell
.\.venv\Scripts\python.exe -m pip check
```

## ▶️ Run the Application

From the project root, run:

```powershell
.\.venv\Scripts\python.exe .\src\SmartFAQSearch.py
```

On the first run, the application may download the embedding model. This requires an internet connection and may take some time.

Enter a question when prompted. Type `exit` to quit.

### Example Queries

**Relevant question**

```text
Where can I find my grades?
```

The application should retrieve the relevant FAQ and display its answer.

**Out-of-scope question**

```text
How do I repair a car?
```

The application should reject the query if its closest match falls outside the configured relevance threshold.

## 🧪 Testing

### Unit tests

Run the unit tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover relevant search results, unrelated questions, empty collections, and clear or ambiguous matches.

**Latest result:** 5 tests passed.

### Real-model evaluation

Run the evaluation script with:

```powershell
.\.venv\Scripts\python.exe .\tests\evaluate_search.py
```

This script uses the real embedding model to evaluate a small, manually selected set of relevant and out-of-scope questions.

## 📊 Evaluation Results

The recorded evaluation used a distance threshold of `0.65`.

| Metric                                         |        Result |
| ---------------------------------------------- | ------------: |
| Relevant queries matched to the expected FAQ   | 13/14 (92.9%) |
| Relevant queries incorrectly rejected          |          0/14 |
| Out-of-scope queries rejected                  |    8/8 (100%) |
| False acceptances in the out-of-scope test set |           0/8 |

### Known Issue

One query—“How do I see my exam results?”—retrieved “When are examination results published?” instead of the expected FAQ, “Where can I find my grades?”

These questions are semantically related, but the result does not match the intended information need. This highlights a limitation of embedding-based retrieval: similar meanings do not always represent the same user intent.

**Important:** These results describe only this small evaluation set. They are preliminary measurements, not a guarantee of performance on unseen questions or real-world university queries.

## 🎯 Learning Outcomes

This project provides practical experience with:

- Text embeddings and semantic similarity
- Vector databases and information retrieval
- Data cleaning and CSV processing
- Relevance thresholds and out-of-scope detection
- Ambiguity handling
- Unit testing and real-model evaluation
- Organizing a Python project into source code, data, and test directories

## ⚠️ Limitations

- The FAQ dataset contains sample data intended for learning and demonstration.
- Semantically similar questions can have different intended meanings.
- The distance threshold and ambiguity margin are heuristic settings that need further evaluation.
- The evaluation dataset is small and manually selected.
- The application is a learning project, not a production-ready university support system.
- The included FAQ data does not represent verified university policies.

## 🔮 Future Improvements

- Expand the FAQ dataset and evaluation questions.
- Improve retrieval for closely related topics such as grades, exam schedules, and result publication dates.
- Evaluate relevance thresholds using a larger, more diverse dataset.
- Add a user interface for easier interaction.
- Explore additional retrieval and ranking strategies.

---

_Built as a learning project to explore semantic search, vector databases, and practical AI engineering._

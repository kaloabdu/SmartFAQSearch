# SmartFAQSearch

A beginner-friendly semantic search application that helps users find relevant answers from university FAQs using natural-language questions.

## Features

- Loads and cleans FAQ data from a CSV file using Pandas.
- Converts FAQ questions into vector embeddings with Sentence Transformers.
- Stores and searches embeddings using ChromaDB.
- Finds relevant answers even when users phrase questions differently from the original FAQs.
- Rejects queries that appear unrelated to the available FAQs.
- Detects potentially ambiguous questions and presents alternatives.
- Includes unit tests and a real-model evaluation script.

## Technologies

- Python 3.12
- Pandas 3.1.0
- Sentence Transformers 6.2.0
- ChromaDB 1.5.9
- unittest

## How It Works

1. The application loads FAQ questions and answers from `data/faqs.csv`.
2. Sentence Transformers converts FAQ questions into numerical embeddings.
3. ChromaDB stores the embeddings and retrieves semantically similar questions.
4. The application compares search distances against a configured threshold.
5. It returns the best answer, asks the user to clarify an ambiguous result, or rejects a query that appears out of scope.

Unlike basic keyword search, semantic search compares the meaning represented by text embeddings. However, similar meanings do not guarantee that the retrieved answer is always correct.

## Project Structure

```text
SmartFAQSearch/
|-- data/
|   `-- faqs.csv
|-- src/
|   `-- main.py
|-- tests/
|   |-- test_search.py
|   `-- evaluate_search.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

The ChromaDB database is generated locally and is excluded from version control.

## Requirements

- Python 3.12
- Internet access for the initial installation and embedding-model download

## Setup on Windows

Open PowerShell in the project root directory.

Create a virtual environment using Python 3.12:

```powershell
py -3.12 -m venv .venv
```

Install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check the installation:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

## Run the Application

```powershell
.\.venv\Scripts\python.exe src\main.py
```

On the first run, the application may download the embedding model, which can take some time.

Follow the prompt to enter a question. Type `exit` to quit.

## Run the Tests

### Unit tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests check relevant search results, unrelated queries, empty collections, and ambiguous or clear answers.

### Real-model evaluation

```powershell
.\.venv\Scripts\python.exe tests\evaluate_search.py
```

This script evaluates search behavior against a small set of manually selected relevant and out-of-scope queries.

## Current Evaluation

Using a distance threshold of `0.65`, the current evaluation produced:

- Relevant queries matched to their expected FAQ: 13/14 (92.9%).
- Relevant queries incorrectly rejected: 0/14.
- Out-of-scope queries rejected: 8/8 (100%).
- False acceptances in the out-of-scope test set: 0/8.

One exam-results query retrieved a related but different FAQ than expected. These results are preliminary: the test set is small and manually selected, so they should not be interpreted as production-level accuracy or a guarantee of performance on unseen questions.

## Learning Goals

This project demonstrates practical concepts in:

- Text embeddings and semantic similarity.
- Vector databases and retrieval.
- Search thresholds and out-of-scope handling.
- Ambiguity handling.
- Unit testing and model evaluation.

## Limitations

- The FAQ dataset contains sample data for learning and demonstration.
- Search results can be incorrect when questions are ambiguous or semantically similar.
- The distance threshold and ambiguity margin are heuristic settings that require further evaluation.
- The application is a learning project, not a production-ready university support system.

**Important:** The included FAQ data does not represent verified university policies.

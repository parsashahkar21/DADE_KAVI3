# Semantic Plagiarism Detection Engine

A Python engine for detecting duplicate and near-duplicate text using locality-sensitive hashing techniques. It supports pairwise comparison, corpus-wide similarity search, and evaluation on labeled question-pair datasets.

## Features

- **MinHash** — Estimates Jaccard similarity over word shingles for near-duplicate detection.
- **SimHash (TF-IDF weighted)** — Builds compact fingerprints and compares them via Hamming distance.
- **LSH index** — Reduces the number of document comparisons in large corpora.
- **CLI** — Compare two files, scan a folder of documents, or evaluate on a CSV of labeled pairs.
- **Bilingual preprocessing** — Normalization and stopword removal for English and Persian text.

## Dataset

This project is designed to be evaluated on the [Quora Question Pairs dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset) from Kaggle.

1. Download the dataset from Kaggle (requires a Kaggle account).
2. Place `questions.csv` in the project root (or pass its path to the CLI).

Expected columns:

| Column        | Description              |
|---------------|--------------------------|
| `question1`   | First question text      |
| `question2`   | Second question text     |
| `is_duplicate`| Label (`0` or `1`)       |

## Installation

```bash
git clone <your-repo-url>
cd dade

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
pip install pytest          # optional, for running tests
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Evaluate on the Quora pairs CSV

```bash
plagiarism-engine pairs \
  --pairs questions.csv \
  --text-col-a question1 \
  --text-col-b question2 \
  --label-col is_duplicate \
  --output output/metrics.csv
```

Use `--limit 1000` to run on a subset for quick experiments.

### Compare two text files

```bash
plagiarism-engine compare \
  --file-a doc_a.txt \
  --file-b doc_b.txt \
  --output output/compare.json
```

### Find similar documents in a folder

Place `.txt` files in a directory, then:

```bash
plagiarism-engine corpus \
  --data path/to/documents \
  --output output/similar_pairs.csv \
  --threshold 0.8
```

## Running tests

```bash
pytest tests/ -v
```

## Project structure

```
dade/
├── src/plagiarism_engine/
│   ├── cli.py           # Command-line interface
│   ├── dataset.py       # Data loading and saving
│   ├── evaluation.py    # Metrics on labeled pairs
│   ├── lsh.py           # LSH index for MinHash signatures
│   ├── minhash.py       # MinHash implementation
│   ├── preprocessing.py # Tokenization and shingling
│   └── simhash.py       # TF-IDF SimHash
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Example results

On the full Quora Question Pairs dataset (~404k pairs):

| Method  | Accuracy | Precision | Recall | F1    |
|---------|----------|-----------|--------|-------|
| MinHash | 64.1%    | 66.6%     | 5.3%   | 9.9%  |
| SimHash | 65.6%    | 59.9%     | 20.7%  | 30.7% |

Results depend on threshold settings in `Evaluator` (`similarity_threshold` for MinHash, `hamming_threshold` for SimHash).

## License

MIT (or update this section to match your chosen license.)

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_text_file(file_path: str) -> str:
    """
    Read a UTF-8 text file.
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_corpus(folder: str) -> Dict[str, str]:
    """
    Load all .txt files from a folder.

    Returns
    -------
    {
        filename: document_text
    }
    """

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(folder)

    corpus = {}

    for file in sorted(folder.glob("*.txt")):
        corpus[file.name] = load_text_file(file)

    return corpus


def corpus_as_lists(folder: str) -> Tuple[List[str], List[str]]:
    """
    Returns

    document_ids
    document_texts
    """

    corpus = load_corpus(folder)

    ids = list(corpus.keys())
    texts = list(corpus.values())

    return ids, texts


def load_pairs_dataset(
    csv_path: str,
    text_col_a: str,
    text_col_b: str,
    label_col: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load Quora / PAN pair dataset.
    """

    read_kwargs = {}
    if limit is not None:
        read_kwargs["nrows"] = limit

    df = pd.read_csv(csv_path, **read_kwargs)

    required_columns = {
        text_col_a,
        text_col_b,
        label_col,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[
        [
            text_col_a,
            text_col_b,
            label_col,
        ]
    ].dropna()

    df = df.rename(
        columns={
            text_col_a: "text_a",
            text_col_b: "text_b",
            label_col: "label",
        }
    )

    return df.reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, output_path: str):
    """
    Save dataframe to csv.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )


def save_json(data, output_path: str):
    """
    Save dictionary as json.
    """

    import json

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )
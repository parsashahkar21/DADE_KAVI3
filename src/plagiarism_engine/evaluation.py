import time
from typing import Dict

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from tqdm import tqdm

from .preprocessing import preprocess_to_shingles, preprocess_to_tokens
from .minhash import MinHasher
from .simhash import TFIDFSimHasher


class Evaluator:

    def __init__(
        self,
        shingle_size: int = 3,
        num_hashes: int = 128,
        similarity_threshold: float = 0.8,
        hamming_threshold: int = 10,
    ):

        self.shingle_size = shingle_size
        self.num_hashes = num_hashes
        self.similarity_threshold = similarity_threshold
        self.hamming_threshold = hamming_threshold

    def evaluate_minhash(self, dataframe: pd.DataFrame) -> Dict:

        minhasher = MinHasher(self.num_hashes)

        y_true = []
        y_pred = []

        start = time.time()

        for _, row in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc="MinHash",
        ):

            shingles_a = preprocess_to_shingles(
                row["text_a"],
                self.shingle_size,
            )

            shingles_b = preprocess_to_shingles(
                row["text_b"],
                self.shingle_size,
            )

            sig_a = minhasher.signature(shingles_a)
            sig_b = minhasher.signature(shingles_b)

            similarity = minhasher.estimate_similarity(
                sig_a,
                sig_b,
            )

            prediction = int(
                similarity >= self.similarity_threshold
            )

            y_true.append(int(row["label"]))
            y_pred.append(prediction)

        elapsed = time.time() - start

        return {
            "Method": "MinHash",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Runtime(sec)": elapsed,
        }

    def evaluate_simhash(self, dataframe: pd.DataFrame):

        token_lists = [
            preprocess_to_tokens(text)
            for text in list(dataframe["text_a"])
            + list(dataframe["text_b"])
        ]

        simhasher = TFIDFSimHasher()

        simhasher.fit(token_lists)

        y_true = []
        y_pred = []

        start = time.time()

        for _, row in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc="SimHash",
        ):

            tokens_a = preprocess_to_tokens(row["text_a"])
            tokens_b = preprocess_to_tokens(row["text_b"])

            hash_a = simhasher.signature(tokens_a)
            hash_b = simhasher.signature(tokens_b)

            distance = simhasher.hamming_distance(
                hash_a,
                hash_b,
            )

            prediction = int(
                distance <= self.hamming_threshold
            )

            y_true.append(int(row["label"]))
            y_pred.append(prediction)

        elapsed = time.time() - start

        return {
            "Method": "SimHash",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Runtime(sec)": elapsed,
        }

    def evaluate(self, dataframe: pd.DataFrame):

        result1 = self.evaluate_minhash(dataframe)
        result2 = self.evaluate_simhash(dataframe)

        return pd.DataFrame(
            [
                result1,
                result2,
            ]
        )
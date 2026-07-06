import hashlib
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List


class TFIDFSimHasher:
    """
    TF-IDF weighted SimHash implementation.

    Each token is hashed to a 64-bit integer.
    For every bit position, token weights are added or subtracted
    depending on whether that bit is 1 or 0.
    """

    def __init__(self, hash_bits: int = 64):
        if hash_bits <= 0:
            raise ValueError("hash_bits must be positive")

        self.hash_bits = hash_bits
        self.idf_: Dict[str, float] = {}

    @staticmethod
    def _stable_hash(value: str) -> int:
        digest = hashlib.md5(value.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def fit(self, documents_tokens: List[List[str]]) -> None:
        document_count = len(documents_tokens)
        document_frequency = defaultdict(int)

        for tokens in documents_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                document_frequency[token] += 1

        self.idf_ = {}

        for token, df in document_frequency.items():
            self.idf_[token] = math.log(
                (1 + document_count) / (1 + df)
            ) + 1.0

    def _token_weight(self, token: str, tf: int) -> float:
        idf = self.idf_.get(token, 1.0)
        return tf * idf

    def signature(self, tokens: Iterable[str]) -> int:
        tokens = list(tokens)

        if not tokens:
            return 0

        token_counts = Counter(tokens)
        vector = [0.0] * self.hash_bits

        for token, tf in token_counts.items():
            token_hash = self._stable_hash(token)
            weight = self._token_weight(token, tf)

            for bit_index in range(self.hash_bits):
                bit_mask = 1 << bit_index

                if token_hash & bit_mask:
                    vector[bit_index] += weight
                else:
                    vector[bit_index] -= weight

        fingerprint = 0

        for bit_index, value in enumerate(vector):
            if value >= 0:
                fingerprint |= 1 << bit_index

        return fingerprint

    def fit_transform(self, documents_tokens: List[List[str]]) -> List[int]:
        self.fit(documents_tokens)
        return [self.signature(tokens) for tokens in documents_tokens]

    @staticmethod
    def hamming_distance(hash_a: int, hash_b: int) -> int:
        return (hash_a ^ hash_b).bit_count()

    def similarity(self, hash_a: int, hash_b: int) -> float:
        distance = self.hamming_distance(hash_a, hash_b)
        return 1.0 - (distance / self.hash_bits)
import hashlib
import random
from typing import Iterable, List, Set


class MinHasher:
    """
    MinHash implementation from scratch.

    Each document is represented as a set of shingles.
    For each hash function, we keep the minimum hash value over all shingles.
    The similarity of two signatures is estimated by the fraction of equal
    components.
    """

    def __init__(
        self,
        num_hashes: int = 128,
        seed: int = 42,
        max_hash: int = (1 << 32) - 1
    ):
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")

        self.num_hashes = num_hashes
        self.seed = seed
        self.max_hash = max_hash

        rng = random.Random(seed)

        self.coefficients_a = [
            rng.randint(1, max_hash - 1)
            for _ in range(num_hashes)
        ]

        self.coefficients_b = [
            rng.randint(0, max_hash - 1)
            for _ in range(num_hashes)
        ]

        self.prime = 4294967311

    @staticmethod
    def _stable_hash(value: str) -> int:
        digest = hashlib.md5(value.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _hash(self, x: int, i: int) -> int:
        return (
            self.coefficients_a[i] * x + self.coefficients_b[i]
        ) % self.prime % self.max_hash

    def signature(self, shingles: Iterable[str]) -> List[int]:
        shingles = set(shingles)

        if not shingles:
            return [self.max_hash] * self.num_hashes

        signature = []

        for i in range(self.num_hashes):
            min_value = self.max_hash

            for shingle in shingles:
                x = self._stable_hash(shingle)
                hashed_value = self._hash(x, i)

                if hashed_value < min_value:
                    min_value = hashed_value

            signature.append(min_value)

        return signature

    @staticmethod
    def estimate_similarity(
        signature_a: List[int],
        signature_b: List[int]
    ) -> float:
        if len(signature_a) != len(signature_b):
            raise ValueError("signatures must have the same length")

        if not signature_a:
            return 0.0

        equal_count = sum(
            1 for a, b in zip(signature_a, signature_b)
            if a == b
        )

        return equal_count / len(signature_a)


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    intersection_size = len(set_a.intersection(set_b))
    union_size = len(set_a.union(set_b))

    return intersection_size / union_size
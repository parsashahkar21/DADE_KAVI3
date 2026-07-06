from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple


class LSHIndex:
    """
    Locality Sensitive Hashing for MinHash signatures.

    The signature is divided into bands.
    Documents that fall into the same bucket in at least one band
    are considered candidate pairs.
    """

    def __init__(self, num_bands: int = 32):
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")

        self.num_bands = num_bands
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = defaultdict(list)
        self.signatures: Dict[str, List[int]] = {}

    def _band_ranges(self, signature_length: int):
        if signature_length % self.num_bands != 0:
            raise ValueError(
                "signature length must be divisible by num_bands"
            )

        rows_per_band = signature_length // self.num_bands

        for band_index in range(self.num_bands):
            start = band_index * rows_per_band
            end = start + rows_per_band
            yield band_index, start, end

    def add(self, doc_id: str, signature: List[int]) -> None:
        self.signatures[doc_id] = signature

        for band_index, start, end in self._band_ranges(len(signature)):
            band_tuple = tuple(signature[start:end])
            bucket_key = (band_index, band_tuple)
            self.buckets[bucket_key].append(doc_id)

    def build(self, signatures: Dict[str, List[int]]) -> None:
        self.buckets.clear()
        self.signatures.clear()

        for doc_id, signature in signatures.items():
            self.add(doc_id, signature)

    def query(self, signature: List[int]) -> Set[str]:
        candidates = set()

        for band_index, start, end in self._band_ranges(len(signature)):
            band_tuple = tuple(signature[start:end])
            bucket_key = (band_index, band_tuple)

            for doc_id in self.buckets.get(bucket_key, []):
                candidates.add(doc_id)

        return candidates

    def candidate_pairs(self) -> Set[Tuple[str, str]]:
        pairs = set()

        for doc_ids in self.buckets.values():
            if len(doc_ids) < 2:
                continue

            for a, b in combinations(sorted(set(doc_ids)), 2):
                pairs.add((a, b))

        return pairs

    def comparison_reduction(self) -> Dict[str, float]:
        n_docs = len(self.signatures)
        all_pairs = n_docs * (n_docs - 1) // 2
        candidate_count = len(self.candidate_pairs())

        if all_pairs == 0:
            reduction = 0.0
        else:
            reduction = 1.0 - (candidate_count / all_pairs)

        return {
            "num_documents": n_docs,
            "all_pairs": all_pairs,
            "candidate_pairs": candidate_count,
            "reduction_ratio": reduction,
        }
import unittest

from plagiarism_engine.preprocessing import (
    preprocess_to_shingles,
    preprocess_to_tokens,
)

from plagiarism_engine.minhash import (
    MinHasher,
    jaccard_similarity,
)

from plagiarism_engine.simhash import (
    TFIDFSimHasher,
)


class TestPreprocessing(unittest.TestCase):

    def test_tokenization(self):

        tokens = preprocess_to_tokens(
            "This is a simple plagiarism detection example."
        )

        self.assertTrue(len(tokens) > 0)

    def test_shingles(self):

        shingles = preprocess_to_shingles(
            "this is a simple plagiarism detection example",
            shingle_size=3,
        )

        self.assertTrue(len(shingles) > 0)


class TestMinHash(unittest.TestCase):

    def test_jaccard(self):

        a = {"a", "b", "c"}
        b = {"b", "c", "d"}

        similarity = jaccard_similarity(a, b)

        self.assertAlmostEqual(
            similarity,
            0.5,
        )

    def test_signature_similarity(self):

        minhasher = MinHasher(128)

        s1 = {
            "this is test",
            "is test file",
            "test file one",
        }

        s2 = {
            "this is test",
            "is test file",
            "test file two",
        }

        sig1 = minhasher.signature(s1)
        sig2 = minhasher.signature(s2)

        similarity = minhasher.estimate_similarity(
            sig1,
            sig2,
        )

        self.assertGreater(
            similarity,
            0.3,
        )


class TestSimHash(unittest.TestCase):

    def test_hamming(self):

        docs = [
            ["machine", "learning"],
            ["machine", "learning", "model"],
        ]

        simhasher = TFIDFSimHasher()

        simhasher.fit(docs)

        h1 = simhasher.signature(docs[0])
        h2 = simhasher.signature(docs[1])

        distance = simhasher.hamming_distance(
            h1,
            h2,
        )

        self.assertLess(
            distance,
            20,
        )


if __name__ == "__main__":
    unittest.main()
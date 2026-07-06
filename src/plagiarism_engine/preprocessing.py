import re
import string
from typing import Iterable, List, Set


ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "while", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "for", "with", "as", "by", "at", "from", "this",
    "that", "these", "those", "it", "its", "into", "about", "than",
    "so", "such", "very", "can", "could", "should", "would", "will",
    "just", "not", "no", "yes", "do", "does", "did", "done"
}

PERSIAN_STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "آن", "را", "با", "برای",
    "است", "بود", "شد", "می", "شود", "های", "ها", "یک", "تا",
    "بر", "هم", "نیز", "اما", "یا", "اگر", "هر", "کرد", "کرده",
    "داشت", "دارد", "دارند", "من", "تو", "او", "ما", "شما", "آنها"
}

STOPWORDS = ENGLISH_STOPWORDS | PERSIAN_STOPWORDS


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    replacements = {
        "ي": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "آ": "ا",
        "\u200c": " ",
        "\u200f": " ",
        "\u200e": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower()

    punctuation = string.punctuation + "،؛؟«»٪×÷ـ"
    text = text.translate(str.maketrans({p: " " for p in punctuation}))

    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    text = normalize_text(text)

    if not text:
        return []

    tokens = text.split()

    tokens = [
        token for token in tokens
        if len(token) > 1 and not token.isnumeric()
    ]

    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]

    return tokens


def word_shingles(tokens: Iterable[str], k: int = 3) -> Set[str]:
    tokens = list(tokens)

    if k <= 0:
        raise ValueError("shingle size must be positive")

    if len(tokens) == 0:
        return set()

    if len(tokens) < k:
        return {" ".join(tokens)}

    shingles = set()
    for i in range(len(tokens) - k + 1):
        shingles.add(" ".join(tokens[i:i + k]))

    return shingles


def preprocess_to_shingles(
    text: str,
    shingle_size: int = 3,
    remove_stopwords: bool = True
) -> Set[str]:
    tokens = tokenize(text, remove_stopwords=remove_stopwords)
    return word_shingles(tokens, k=shingle_size)


def preprocess_to_tokens(
    text: str,
    remove_stopwords: bool = True
) -> List[str]:
    return tokenize(text, remove_stopwords=remove_stopwords)
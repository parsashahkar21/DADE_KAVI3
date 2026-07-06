import click
import pandas as pd

from .dataset import (
    load_text_file,
    load_corpus,
    load_pairs_dataset,
    save_dataframe,
    save_json,
)

from .preprocessing import (
    preprocess_to_shingles,
    preprocess_to_tokens,
)

from .minhash import MinHasher
from .simhash import TFIDFSimHasher
from .evaluation import Evaluator


@click.group()
def cli():
    """Semantic Plagiarism Detection Engine"""
    pass


# ---------------------------------------------------------
# Compare Two Files
# ---------------------------------------------------------
@cli.command()
@click.option("--file-a", required=True, help="First text file")
@click.option("--file-b", required=True, help="Second text file")
@click.option("--output", required=True)
@click.option("--shingle-size", default=3)
@click.option("--num-hashes", default=128)
def compare(file_a, file_b, output, shingle_size, num_hashes):

    text_a = load_text_file(file_a)
    text_b = load_text_file(file_b)

    shingles_a = preprocess_to_shingles(
        text_a,
        shingle_size,
    )

    shingles_b = preprocess_to_shingles(
        text_b,
        shingle_size,
    )

    minhasher = MinHasher(num_hashes)

    sig_a = minhasher.signature(shingles_a)
    sig_b = minhasher.signature(shingles_b)

    minhash_similarity = minhasher.estimate_similarity(
        sig_a,
        sig_b,
    )

    simhasher = TFIDFSimHasher()

    tokens_a = preprocess_to_tokens(text_a)
    tokens_b = preprocess_to_tokens(text_b)

    simhasher.fit(
        [
            tokens_a,
            tokens_b,
        ]
    )

    hash_a = simhasher.signature(tokens_a)
    hash_b = simhasher.signature(tokens_b)

    simhash_similarity = simhasher.similarity(
        hash_a,
        hash_b,
    )

    result = {
        "minhash_similarity": round(
            minhash_similarity,
            4,
        ),
        "simhash_similarity": round(
            simhash_similarity,
            4,
        ),
    }

    save_json(result, output)

    click.echo(result)


# ---------------------------------------------------------
# Search Similar Documents in Folder
# ---------------------------------------------------------
@cli.command()
@click.option("--data", required=True)
@click.option("--output", required=True)
@click.option("--threshold", default=0.8)
@click.option("--shingle-size", default=3)
@click.option("--num-hashes", default=128)
def corpus(
    data,
    output,
    threshold,
    shingle_size,
    num_hashes,
):

    corpus = load_corpus(data)

    minhasher = MinHasher(num_hashes)

    signatures = {}

    for filename, text in corpus.items():

        shingles = preprocess_to_shingles(
            text,
            shingle_size,
        )

        signatures[filename] = minhasher.signature(
            shingles
        )

    rows = []

    files = list(signatures.keys())

    for i in range(len(files)):
        for j in range(i + 1, len(files)):

            similarity = minhasher.estimate_similarity(
                signatures[files[i]],
                signatures[files[j]],
            )

            if similarity >= threshold:

                rows.append(
                    {
                        "doc_a": files[i],
                        "doc_b": files[j],
                        "similarity": similarity,
                    }
                )

    save_dataframe(
        pd.DataFrame(rows),
        output,
    )

    click.echo(
        f"{len(rows)} candidate pairs found."
    )


# ---------------------------------------------------------
# Evaluate on Pair Dataset
# ---------------------------------------------------------
@cli.command()
@click.option("--pairs", required=True)
@click.option("--text-col-a", required=True)
@click.option("--text-col-b", required=True)
@click.option("--label-col", required=True)
@click.option("--limit", default=None, type=int)
@click.option("--output", required=True)
def pairs(
    pairs,
    text_col_a,
    text_col_b,
    label_col,
    limit,
    output,
):

    dataframe = load_pairs_dataset(
        pairs,
        text_col_a,
        text_col_b,
        label_col,
        limit,
    )

    evaluator = Evaluator()

    metrics = evaluator.evaluate(
        dataframe
    )

    save_dataframe(
        metrics,
        output,
    )

    click.echo(metrics)


if __name__ == "__main__":
    cli()
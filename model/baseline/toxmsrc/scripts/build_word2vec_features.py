#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import gensim.models
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Build ToxMSRC Word2Vec features from a labeled CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV file.")
    parser.add_argument(
        "--sequence-col",
        default="sequence",
        help="Column name containing peptide sequences.",
    )
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Output .npz path. Saved array key is 'x'.",
    )
    parser.add_argument(
        "--word2vec-model",
        default=str(project_dir / "word2vec_model" / "word2vec.model"),
        help="Path to the trained gensim Word2Vec model.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=50,
        help="Fixed sequence length used by ToxMSRC.",
    )
    parser.add_argument(
        "--kmer-len",
        type=int,
        default=2,
        help="k-mer length used to build Word2Vec tokens.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride used to slide over each sequence.",
    )
    return parser.parse_args()


def normalize_sequence(sequence: str, target_len: int, padding_residue: str = "X") -> str:
    sequence = str(sequence).strip().upper()
    if len(sequence) >= target_len:
        return sequence[:target_len]
    return sequence + (padding_residue * (target_len - len(sequence)))


def generate_kmers(sequence: str, kmer_len: int, stride: int) -> list[str]:
    return [
        sequence[idx:idx + kmer_len]
        for idx in range(0, (len(sequence) - kmer_len) + 1, stride)
    ]


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_npz = Path(args.output_npz).resolve()
    word2vec_model = Path(args.word2vec_model).resolve()

    df = pd.read_csv(input_csv)
    if args.sequence_col not in df.columns:
        raise ValueError(
            f"Column '{args.sequence_col}' not found in {input_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    w2v = gensim.models.Word2Vec.load(str(word2vec_model))
    vector_size = w2v.wv.vector_size
    feature_len = ((args.max_len - args.kmer_len) // args.stride) + 1

    features = np.zeros((len(df), feature_len, vector_size), dtype=np.float32)
    unknown_kmers = 0

    for row_idx, raw_sequence in enumerate(df[args.sequence_col].fillna("").astype(str)):
        sequence = normalize_sequence(raw_sequence, target_len=args.max_len)
        kmers = generate_kmers(sequence, kmer_len=args.kmer_len, stride=args.stride)
        if len(kmers) != feature_len:
            raise ValueError(
                f"Unexpected k-mer count for row {row_idx}: got {len(kmers)}, "
                f"expected {feature_len}."
            )
        for token_idx, token in enumerate(kmers):
            if token in w2v.wv:
                features[row_idx, token_idx] = w2v.wv[token]
            else:
                unknown_kmers += 1

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, x=features)

    print(f"input_csv={input_csv}")
    print(f"output_npz={output_npz}")
    print(f"shape={features.shape}")
    print(f"vector_size={vector_size}")
    print(f"unknown_kmers={unknown_kmers}")


if __name__ == "__main__":
    main()

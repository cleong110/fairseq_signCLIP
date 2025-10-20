#!/usr/bin/env python3
"""
Embed unique text values from a CSV column and save them as a Parquet dataset.

Each unique string from the specified CSV column is embedded using the given
text embedding model and saved along with the original text, embedding, and
metadata.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from search_with_text_and_poses import embed_text

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------
def embed_texts_from_csv(
    *,
    csv_path: Path,
    text_column: str,
    embedding_model: str,
    output_dir: Path,
) -> None:
    """
    Embed unique text entries from a CSV column and save results to a Parquet dataset.

    Parameters:
        csv_path: Path to the input CSV file.
        text_column: Column name containing text to embed.
        embedding_model: Name of the embedding model to use.
        output_dir: Directory where embeddings will be saved.
    """
    if not csv_path.is_file():
        msg = f"CSV file not found: {csv_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        msg = f"Column '{text_column}' not found in CSV: {csv_path}"
        logger.error(msg)
        raise ValueError(msg)

    unique_texts = sorted(df[text_column].dropna().unique())
    logger.info(
        f"Loaded {len(df)} rows from {csv_path.name}; {len(unique_texts)} unique text values to embed."
    )

    rows = []
    case_variants = {
        "lower": str.lower,
        "upper": str.upper,
        "title": str.title,
    }

    prefix_variants = {
        "none": lambda t: t,
        "en_ase": lambda t: f"<en> <ase> {t}",
    }

    for text in tqdm(unique_texts, desc=f"Embedding texts with {embedding_model}"):
        for case_name, case_func in case_variants.items():
            variant = case_func(text)

            for prefix_name, prefix_func in prefix_variants.items():
                text_to_embed = prefix_func(variant)
                embedding = embed_text(
                    text_to_embed, model_name=embedding_model
                ).squeeze()

                rows.append(
                    {
                        "text_original": text,
                        "text_variant": variant,
                        "case_type": case_name,
                        "prefix_type": prefix_name,
                        "text_embedded": text_to_embed,
                        "embedding": embedding.tolist(),
                        "embedding_model": embedding_model,
                    }
                )

    if not rows:
        logger.info("No embeddings generated — nothing to save.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(pd.DataFrame(rows))
    pq.write_to_dataset(
        table,
        root_path=output_dir,
        partition_cols=["embedding_model", "case_type", "prefix_type"],
    )

    logger.info(
        f"Saved {len(rows)} text embeddings partitioned by model and case type and prefix to {output_dir.resolve()}"
    )


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed unique text values from a CSV column."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Name of the column containing text to embed.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="asl_finetune_checkpoint_best",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("embeddings_texts"),
        help="Output directory for Parquet dataset.",
    )

    args = parser.parse_args()

    embed_texts_from_csv(
        csv_path=args.csv_path,
        text_column=args.text_column,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

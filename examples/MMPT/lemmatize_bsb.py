from __future__ import annotations

import argparse
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm import tqdm

from sign_language_gloss_utils.glosses.text_utils import preprocess_text


def process_line(line: str) -> list[str]:
    """
    Preprocess a single line into lemmas using sign_language_gloss_utils.
    """
    return preprocess_text(line)


def load_lines(file_path: Path) -> list[str]:
    """
    Load a UTF-8 text file into a list of lines.
    """
    with file_path.open("r", encoding="utf-8") as f:
        return f.read().splitlines()


def count_lemmas_parallel(lines: Sequence[str]) -> Counter[str]:
    """
    Run preprocess_text in parallel across all lines and aggregate lemma counts.
    """
    counter: Counter[str] = Counter()
    with ProcessPoolExecutor() as executor:
        # Using map is efficient for large inputs; tqdm wraps it for progress.
        for line_lemmas in tqdm(
            executor.map(process_line, lines, chunksize=100),
            total=len(lines),
            desc="Processing lines",
        ):
            counter.update(line_lemmas)
    return counter


def save_counts_to_csv(counts: Counter[str], output_path: Path) -> None:
    """
    Save lemma counts to CSV with columns: lemma, count.
    """
    df = pd.DataFrame(counts.items(), columns=["lemma", "count"]).sort_values(
        by="count", ascending=False, ignore_index=True
    )
    df.to_csv(output_path, index=False, encoding="utf-8")


def main() -> None:
    """
    Parallel preprocessing of text lines to extract and count lemmas, saving as CSV.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Parallel preprocessing of text corpus to extract lemma counts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eng-engbsb.txt"),
        help="Input text file (default: eng-engbsb.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: input_file_stem + '_lemmas.csv')",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        logger.error("Input file %s not found.", input_path)
        return

    output_path: Path = (
        args.output if args.output else input_path.with_name(f"{input_path.stem}_lemmas.csv")
    )

    logger.info("Loading lines from %s", input_path)
    lines = load_lines(input_path)
    logger.info("Loaded %d lines", len(lines))

    logger.info("Starting parallel preprocessing...")
    lemma_counts = count_lemmas_parallel(lines)

    logger.info("Saving %d unique lemmas to %s", len(lemma_counts), output_path)
    save_counts_to_csv(lemma_counts, output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()

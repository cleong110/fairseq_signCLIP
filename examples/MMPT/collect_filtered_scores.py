#!/usr/bin/env python3
"""
Collect evaluation stats CSVs, extract metadata from their paths,
and combine into a single deduplicated DataFrame.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_metadata(path: Path) -> dict[str, Any]:
    """
    Extract metadata from a given CSV file path.
    Expected patterns inside the path include:
      - samplespergloss_{int}
      - windowsize{int}_step{int}
      - cbt-{int}
      - prominence_{float}_heightmult_{float}
    """
    # /opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/results/asl_finetune_checkpoint_best/samplespergloss_0/start_0_end_None/windowsize500_step100/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-001-ase-3-passage _ god creates the world.pose-mediapipe/seg_idx9999/peaks/prominence_1.0_heightmult_0.0/DAY/all_scores_DAY_peaks_eval_stats_filtered.csv
    text = str(path)

    # samples per gloss
    spg_match = re.search(r"samplespergloss_(\d+)", text)
    samples_per_gloss = int(spg_match.group(1)) if spg_match else None

    # window size and step
    win_match = re.search(r"windowsize(\d+)_step(\d+)", text)
    windowsize = int(win_match.group(1)) if win_match else None
    step = int(win_match.group(2)) if win_match else None

    # CBT number
    cbt_match = re.search(r"cbt-(\d+)", text, flags=re.IGNORECASE)
    cbt = int(cbt_match.group(1)) if cbt_match else None

    # prominence and heightmult
    prom_match = re.search(r"prominence_([\d.]+)_heightmult_([\d.]+)", text)
    prominence = float(prom_match.group(1)) if prom_match else None
    heightmult = float(prom_match.group(2)) if prom_match else None

    gloss_label_count = len(path.parent.name.split("_"))
    threshold_strat = "dynamic" if gloss_label_count == 1 else "average"

    return {
        "samples_per_gloss": samples_per_gloss,
        "windowsize_ms": windowsize,
        "step_ms": step,
        "cbt": cbt,
        "prominence": prominence,
        "heightmult": heightmult,
        "gloss_label_count": gloss_label_count,
    }


def collect_csvs(root: Path) -> pd.DataFrame:
    """
    Recursively collect all CSVs under root, parse metadata,
    and combine into a single DataFrame.
    """
    csv_files = list(root.rglob("*_eval_stats_filtered.csv"))
    logger.info("Found %d CSV files", len(csv_files))

    all_dfs: list[pd.DataFrame] = []

    for path in tqdm(csv_files, desc="Reading CSVs"):
        if "seg_idx9999" not in str(path.resolve()):
            print(f"{path} not in seg_idx9999")
            continue
        try:
            df = pd.read_csv(path)

            # Remove rows where query_label is "TOTAL"
            # df = df[df["query_label"] != "TOTAL"]

            # add path
            df["path"] = str(path)

            df["parent_name"] = str(path.parent.name)
            df["grandparent_name"] = str(path.parent.parent.name)
            df["grandparent_path"] = str(path.parent.parent)

            if "NoEnglish" in str(path):
                df["English"] = False
            else:
                df["English"] = True

            if "_flipped" in str(path):
                df["Flipped"] = True
            else:
                df["Flipped"] = False

            metadata = extract_metadata(path)
            for key, val in metadata.items():
                df[key] = val
            all_dfs.append(df)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping %s due to error: %s", path, e)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"{len(combined)} combined rows before deduping")
    deduped = combined.drop_duplicates()
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation stats CSVs with metadata parsing."
    )
    parser.add_argument("root", type=Path, help="Root directory to search")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results_analysis/combined.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = collect_csvs(args.root)
    if df.empty:
        logger.warning("No valid CSVs found.")
        return

    df.to_csv(args.output, index=False)
    logger.info("Wrote combined CSV to %s with %d rows", args.output, len(df))


if __name__ == "__main__":
    main()

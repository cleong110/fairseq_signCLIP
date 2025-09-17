#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def recalc_metrics(group: pd.DataFrame) -> pd.Series:
    """Recalculate totals and metrics for a group of rows."""
    preds_count = group["predictions_count"].sum()
    ground_truth_count = group["ground_truth_count"].sum()
    tp = group["true_positives"].sum()
    fp = group["false_positives"].sum()
    fn = group["false_negatives"].sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    logger.debug(
        "Group %s -> TP=%d FP=%d FN=%d precision=%.4f recall=%.4f f1=%.4f",
        group.name if hasattr(group, "name") else "UNKNOWN",
        tp,
        fp,
        fn,
        precision,
        recall,
        f1,
    )

    

    
    results = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions_count":preds_count,
            "ground_truth_count":ground_truth_count,
        }
    

    for col in group.columns:
        if len(group[col].unique())==1 and col not in results:
            results[col] = group[col].unique()[0]

    return pd.Series(results)


def process_file(csv_path: Path) -> pd.DataFrame:
    """Read CSV, filter, group, and recalculate metrics."""
    logger.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    logger.info("Initial rows: %d", len(df))



    # Apply filters
    df_filtered = df[
        (df["gloss_label_count"] == 1) & (df["query_label"] != "TOTAL")
    ]
    logger.info("Filtered rows: %d", len(df_filtered))

    if df_filtered.empty:
        logger.warning("No rows after filtering")
        return pd.DataFrame()

    # Group and recalc
    results = []
    for grandparent_path, group in tqdm(
        df_filtered.groupby("grandparent_path"), desc="Processing groups"
    ):
        metrics = recalc_metrics(group)
        metrics["grandparent_path"] = grandparent_path
        assert len(group["cbt"].unique()) == 1
        metrics["cbt"]=group["cbt"].unique()[0]
        results.append(metrics)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from combined.csv"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to combined.csv file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to save aggregated CSV",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    results = process_file(args.csv_path)

    if results.empty:
        logger.info("No results to output")
        return

    if args.output:
        results.to_csv(args.output, index=False)
        logger.info("Saved results to %s", args.output)
    else:
        logger.info("Aggregated results:\n%s", results.to_string(index=False))


if __name__ == "__main__":
    main()

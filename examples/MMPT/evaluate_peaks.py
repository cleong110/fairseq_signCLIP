from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


def time_str_to_ms(time_str: str) -> int:
    """Convert mm:ss to milliseconds."""
    minutes, seconds = map(int, time_str.strip().split(":"))
    return (minutes * 60 + seconds) * 1000


def load_ground_truth(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_ms"] = df["Time (minutes:sec)"].map(time_str_to_ms)
    return df


def load_predictions(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "window_midpoint_ms" not in df.columns or "query_label" not in df.columns:
        raise ValueError(
            "Predictions file must have 'query_label' and 'window_midpoint_ms' columns."
        )
    return df


def match_predictions(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    window: int = 1000,
) -> pd.DataFrame:
    """Match predictions to ground truth within a ±window (in ms)."""
    gt_by_sign = {sign: ts.values for sign, ts in gt_df.groupby("Sign")["timestamp_ms"]}

    results = []
    for row in tqdm(
        preds_df.itertuples(index=False),
        total=len(preds_df),
        desc="Matching predictions",
    ):
        label = row.query_label
        ts = row.window_midpoint_ms
        matches = gt_by_sign.get(label, [])
        correct = any(abs(ts - gt_ts) <= window for gt_ts in matches)
        results.append(correct)

    preds_df = preds_df.copy()
    preds_df["correct"] = results
    return preds_df

def compute_eval_stats(
    evaluated_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    window: int = 1000,
) -> pd.DataFrame:
    """Compute per-label precision, recall, and F1 score."""
    gt_by_label = gt_df.groupby("Sign")["timestamp_ms"].apply(list).to_dict()
    pred_by_label = evaluated_df.groupby("query_label")

    all_labels = set(gt_by_label) | set(pred_by_label.groups)

    stats = []
    for label in sorted(all_labels):
        gt_timestamps = gt_by_label.get(label, [])
        pred_rows = (
            pred_by_label.get_group(label)
            if label in pred_by_label.groups
            else pd.DataFrame(columns=evaluated_df.columns)
        )
        pred_timestamps = pred_rows["window_midpoint_ms"].to_numpy()
        correct_mask = pred_rows["correct"].to_numpy() if not pred_rows.empty else []

        tp = np.sum(correct_mask) if len(correct_mask) > 0 else 0
        fp = len(pred_rows) - tp
        fn = len(gt_timestamps)

        matched_gt = set()
        for pred_ts in pred_timestamps[correct_mask]:
            for gt_ts in gt_timestamps:
                if abs(pred_ts - gt_ts) <= window:
                    matched_gt.add(gt_ts)
                    break
        fn -= len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        stats.append(
            {
                "query_label": label,
                "predictions_count": len(pred_rows),
                "ground_truth_count": len(gt_timestamps),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    df = pd.DataFrame(stats)

    # --- Append TOTAL row ---
    total_tp = df["true_positives"].sum()
    total_fp = df["false_positives"].sum()
    total_fn = df["false_negatives"].sum()
    total_preds = df["predictions_count"].sum()
    total_gt = df["ground_truth_count"].sum()

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0

    df.loc[len(df)] = {
        "query_label": "TOTAL",
        "predictions_count": total_preds,
        "ground_truth_count": total_gt,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(total_precision, 4),
        "recall": round(total_recall, 4),
        "f1": round(total_f1, 4),
    }

    return df



def main(
    ground_truth_csv: Path,
    predictions_csv: Path,
    output_csv: Path | None = None,
) -> None:
    logging.info("Loading ground truth from %s", ground_truth_csv)
    gt_df = load_ground_truth(ground_truth_csv)

    logging.info("Loading predictions from %s", predictions_csv)
    preds_df = load_predictions(predictions_csv)

    logging.info("Evaluating predictions...")
    evaluated_df = match_predictions(preds_df, gt_df)

    total = len(evaluated_df)
    correct = evaluated_df["correct"].sum()
    accuracy = correct / total if total > 0 else 0.0
    logging.info(
        "Correct predictions: %d / %d (%.2f%%)", correct, total, accuracy * 100
    )

    if output_csv is None:
        suffix = predictions_csv.suffix
        base = predictions_csv.stem
        output_csv = predictions_csv.with_name(f"{base}_evaluated{suffix}")

    logging.info("Saving evaluated predictions to %s", output_csv.resolve())
    evaluated_df.to_csv(output_csv, index=False)

    # --- Compute Full Stats ---
    eval_stats_df = compute_eval_stats(evaluated_df, gt_df)
    stats_output_csv = output_csv.with_name(
        output_csv.stem.replace("_evaluated", "_eval_stats") + ".csv"
    )
    logging.info("Saving full evaluation stats to %s", stats_output_csv.resolve())
    eval_stats_df.to_csv(stats_output_csv, index=False)

    # --- Filtered by Folder Labels ---
    folder_labels = set(predictions_csv.parent.name.split("_"))
    filtered_eval_df = evaluated_df[evaluated_df["query_label"].isin(folder_labels)]
    filtered_gt_df = gt_df[gt_df["Sign"].isin(folder_labels)]
    filtered_df = compute_eval_stats(filtered_eval_df, filtered_gt_df)
    filtered_output_csv = stats_output_csv.with_name(
        stats_output_csv.stem.replace("_eval_stats", "_eval_stats_filtered") + ".csv"
    )
    logging.info(
        "Saving filtered evaluation stats to %s", filtered_output_csv.resolve()
    )
    filtered_df.to_csv(filtered_output_csv, index=False)

    # --- Predicted-only Labels ---
    predicted_labels = set(evaluated_df["query_label"])
    predicted_eval_df = evaluated_df[evaluated_df["query_label"].isin(predicted_labels)]
    predicted_gt_df = gt_df[gt_df["Sign"].isin(predicted_labels)]
    predicted_only_df = compute_eval_stats(predicted_eval_df, predicted_gt_df)
    predicted_output_csv = stats_output_csv.with_name(
        stats_output_csv.stem.replace("_eval_stats", "_eval_stats_predicted") + ".csv"
    )
    logging.info(
        "Saving predicted-only evaluation stats to %s", predicted_output_csv.resolve()
    )
    predicted_only_df.to_csv(predicted_output_csv, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Evaluate predictions against ground truth timestamps."
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        required=True,
        help="Path to ground truth .csv (tab-separated)",
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Path to predictions .csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to save the output with 'correct' column",
    )
    args = parser.parse_args()

    main(
        ground_truth_csv=args.ground_truth,
        predictions_csv=args.predictions,
        output_csv=args.output,
    )

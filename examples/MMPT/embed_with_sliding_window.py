import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from search_with_text_and_poses import (
    embed_pose,
    load_pose,
    get_sliding_window_frame_ranges,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_expected_windows(
    start_time_ms: int, end_time_ms: int, window_size_ms: int, step_size_ms: int
) -> set[tuple[int, int]]:
    """Return set of all expected (window_start, window_end) pairs."""
    return {
        (w_start, w_end)
        for w_start, w_end in get_sliding_window_frame_ranges(
            max=end_time_ms,
            window_length=window_size_ms,
            stride=step_size_ms,
            start=start_time_ms,
            end=end_time_ms,
        )
    }


def embed_with_sliding_window(
    pose_path: Path,
    start_time_ms: int,
    end_time_ms: int,
    window_size_ms: int,
    step_size_ms: int,
    model_name: str,
    output_dir: Path,
) -> None:
    """Embed pose into sliding windows and save parquet with hive partitioning."""

    expected_windows = get_expected_windows(
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        window_size_ms=window_size_ms,
        step_size_ms=step_size_ms,
    )

    # Partition path for this model & window_size
    partition_dir = (
        output_dir / f"model={model_name}" / f"window_size_ms={window_size_ms}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)

    existing_windows: set[tuple[int, int]] = set()
    parquet_files = list(partition_dir.glob("*.parquet"))
    if parquet_files:
        df_existing = pd.concat(
            [
                pd.read_parquet(
                    pf, columns=["window_start_ms", "window_end_ms"], engine="pyarrow"
                )
                for pf in parquet_files
            ],
            ignore_index=True,
        )
        existing_windows = set(
            zip(df_existing["window_start_ms"], df_existing["window_end_ms"])
        )

    missing_windows = expected_windows - existing_windows
    if not missing_windows:
        logger.info(
            f"All {len(expected_windows)} windows already embedded for model={model_name}, "
            f"window_size_ms={window_size_ms}"
        )
        return

    logger.info(
        f"Computing {len(missing_windows)} missing windows "
        f"(out of {len(expected_windows)}) for {pose_path.name}"
    )

    results = []
    for window_start, window_end in tqdm(
        sorted(missing_windows),
        desc=f"Embedding windows from {pose_path.name}",
    ):
        pose = load_pose(
            pose_path=pose_path, start_time_ms=window_start, end_time_ms=window_end
        )
        embedding = embed_pose(pose, model_name=model_name).squeeze()

        results.append(
            {
                "window_start_ms": window_start,
                "window_end_ms": window_end,
                "window_midpoint_ms": (window_start + window_end) // 2,
                "embedding": embedding,
                "model": model_name,
                "window_size_ms": window_size_ms,
            }
        )

    df = pd.DataFrame(results)

    emb_dim = df["embedding"].iloc[0].shape[0]
    emb_cols = [f"embedding_{i}" for i in range(emb_dim)]
    emb_matrix = np.stack(df["embedding"].to_numpy())
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)

    out_df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)

    out_file = partition_dir / f"{pose_path.stem}_part.parquet"
    out_df.to_parquet(out_file, index=False, engine="pyarrow")

    logger.info(f"Saved {len(out_df)} embeddings → {out_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute SignCLIP pose embeddings over sliding windows."
    )
    parser.add_argument("pose_path", type=Path)
    parser.add_argument("--start_time_ms", type=int, default=0)
    parser.add_argument("--end_time_ms", type=int, default=None)
    parser.add_argument("--window_size_ms", type=int, default=1000)
    parser.add_argument("--step_size_ms", type=int, default=100)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--output_dir", type=Path, default=None)

    args = parser.parse_args()

    if args.pose_path.is_dir():
        pose_paths = list(pose_path.rglob("*.pose"))
    for pose_path in pose_paths:
        full_pose = load_pose(pose_path)
        full_duration_ms = int(1000 * full_pose.body.data.shape[0] / full_pose.body.fps)

        end_time = args.end_time_ms or full_duration_ms

        # Default output_dir: same folder, same basename, with `.window_embeddings`
        if args.output_dir is None:
            output_dir = pose_path.with_suffix("")  # strip extension
            output_dir = output_dir.with_name(output_dir.name + ".window_embeddings")
        else:
            output_dir = args.output_dir

        embed_with_sliding_window(
            pose_path=pose_path,
            start_time_ms=args.start_time_ms,
            end_time_ms=end_time,
            window_size_ms=args.window_size_ms,
            step_size_ms=args.step_size_ms,
            model_name=args.model,
            output_dir=output_dir,
        )

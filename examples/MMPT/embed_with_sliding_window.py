import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
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


def get_existing_windows(
    dataset_dir: Path, model: str, window_size_ms: int, columns=None
) -> set[tuple[int, int]]:
    """Scan existing Hive-partitioned dataset for completed windows."""
    if not dataset_dir.exists():
        return set()

    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive")
    filt = (ds.field("model") == model) & (ds.field("window_size_ms") == window_size_ms)

    table = dataset.to_table(filter=filt, columns=columns)
    df = table.to_pandas()
    return df


def get_existing_windows_set(
    dataset_dir: Path, model: str, window_size_ms: int
) -> set[tuple[int, int]]:
    """Scan existing Hive-partitioned dataset for completed windows."""
    if not dataset_dir.exists():
        return set()

    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive")
    filt = (ds.field("model") == model) & (ds.field("window_size_ms") == window_size_ms)

    table = dataset.to_table(filter=filt, columns=["window_start_ms", "window_end_ms"])
    df = table.to_pandas()
    return set(zip(df["window_start_ms"], df["window_end_ms"]))


def embed_with_sliding_window(
    pose_path: Path,
    start_time_ms: int,
    end_time_ms: int,
    window_size_ms: int,
    step_size_ms: int,
    model_name: str,
    output_dir: Path,
) -> None:
    """Embed pose into sliding windows and save parquet with hive partitioning (embedding column stays as list)."""
    logger.info(f"Calculating missing windows from {output_dir}")

    expected_windows = get_expected_windows(
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        window_size_ms=window_size_ms,
        step_size_ms=step_size_ms,
    )

    existing_windows = get_existing_windows_set(
        dataset_dir=output_dir, model=model_name, window_size_ms=window_size_ms
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

    rows = []
    for window_start, window_end in tqdm(
        sorted(missing_windows),
        desc=f"Embedding windows from {pose_path.name}",
    ):
        pose = load_pose(
            pose_path=pose_path, start_time_ms=window_start, end_time_ms=window_end
        )
        embedding = embed_pose(pose, model_name=model_name).squeeze()

        rows.append(
            {
                "window_start_ms": window_start,
                "window_end_ms": window_end,
                "window_midpoint_ms": (window_start + window_end) // 2,
                "embedding": embedding.tolist(),  # <-- keep as list, not expanded
                "model": model_name,
                "window_size_ms": window_size_ms,
            }
        )

    df = pd.DataFrame(rows)

    df.to_parquet(
        output_dir,
        index=False,
        engine="pyarrow",
        partition_cols=["model", "window_size_ms"],
    )

    logger.info(f"Saved {len(df)} new embeddings to dataset at {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute SignCLIP pose embeddings over sliding windows."
    )
    parser.add_argument("pose_path", type=Path)
    parser.add_argument("--start_time_ms", type=int, default=0)
    parser.add_argument("--end_time_ms", type=int, default=None)
    parser.add_argument("--window_size_ms", type=int, default=500)
    parser.add_argument("--step_size_ms", type=int, default=100)
    parser.add_argument("--model", type=str, default="asl_finetune_checkpoint_best")
    parser.add_argument("--output_dir", type=Path, default=None)

    args = parser.parse_args()

    full_pose = load_pose(args.pose_path)
    full_duration_ms = int(1000 * full_pose.body.data.shape[0] / full_pose.body.fps)

    end_time = args.end_time_ms or full_duration_ms

    # Default: same basename, new `.window_embeddings` dir
    if args.output_dir is None:
        output_dir = args.pose_path.with_suffix("")
        output_dir = output_dir.with_name(output_dir.name + ".window_embeddings")
    else:
        output_dir = args.output_dir
    # if args.window_size_ms is None:
    #     for window_size_ms in [500, 1000, 3000]:
    #         embed_with_sliding_window(
    #               pose_path=args.pose_path,
    #               start_time_ms=args.start_time_ms,
    #               end_time_ms=end_time,
    #               window_size_ms=window_size_ms,
    #               step_size_ms=args.step_size_ms,
    #               model_name=args.model,
    #               output_dir=output_dir,
    #           )
    embed_with_sliding_window(
        pose_path=args.pose_path,
        start_time_ms=args.start_time_ms,
        end_time_ms=end_time,
        window_size_ms=args.window_size_ms,
        step_size_ms=args.step_size_ms,
        model_name=args.model,
        output_dir=output_dir,
    )

    # existing_windows_sample = get_existing_windows(
    #     dataset_dir=output_dir, model=args.model, window_size_ms=args.window_size_ms
    # ).sample(5)
    # logger.info(f"Sample: {existing_windows_sample}")
    # logger.info(f"Sample info: {existing_windows_sample.info()}")
    # for embed in existing_windows_sample["embedding"].to_list():
    #     logger.info(f"type: {type(embed)}: {embed.shape}")

import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from search_with_text_and_poses import (
    embed_pose,
    load_pose,
    get_sliding_window_frame_ranges,
)


def embed_with_sliding_window(
    pose_path: Path,
    start_time_ms: int,
    end_time_ms: int,
    window_size_ms: int,
    step_size_ms: int,
    model_name: str,
    output_path: Path,
):
    results = []
    pose_windows_list = []
    for window_start, window_end in tqdm(
        get_sliding_window_frame_ranges(
            max=end_time_ms,
            window_length=window_size_ms,
            stride=step_size_ms,
            start=start_time_ms,
            end=end_time_ms,
        ),
        desc=f"Embedding windows from {pose_path.name}",
    ):
        pose = load_pose(
            pose_path=pose_path, start_time_ms=window_start, end_time_ms=window_end
        ) # how about we accumulate in a list
        pose_windows_list.append(pose)

    try:
        embedding = embed_pose(pose, model_name=model_name).squeeze()
    except Exception as e:
        print(f"Embedding failed for window {window_start}-{window_end}: {e}")
        continue

        results.append({
            "window_start_ms": window_start,
            "window_end_ms": window_end,
            "window_midpoint_ms": (window_start + window_end) // 2,
            "embedding": embedding,
        })

    # Save to parquet with vector unpacked to columns
    df = pd.DataFrame(results)
    emb_dim = df["embedding"].iloc[0].shape[0]
    emb_cols = [f"embedding_{i}" for i in range(emb_dim)]
    emb_matrix = np.stack(df["embedding"].to_numpy())
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)
    out_df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"Saved {len(out_df)} embeddings to {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute SignCLIP pose embeddings over sliding windows."
    )
    parser.add_argument("pose_path", type=Path)
    parser.add_argument("--start_time_ms", type=int, default=0)
    parser.add_argument("--end_time_ms", type=int, default=None)
    parser.add_argument("--window_size_ms", type=int, default=1000)
    parser.add_argument("--step_size_ms", type=int, default=500)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--output_path", type=Path, default=None)

    args = parser.parse_args()

    full_pose = load_pose(args.pose_path)
    full_duration_ms = int(1000 * full_pose.body.data.shape[0] / full_pose.body.fps)

    end_time = args.end_time_ms or full_duration_ms
    output_path = (
        args.output_path
        or args.pose_path.with_suffix(".window_embeddings.parquet")
    )

    embed_with_sliding_window(
        pose_path=args.pose_path,
        start_time_ms=args.start_time_ms,
        end_time_ms=end_time,
        window_size_ms=args.window_size_ms,
        step_size_ms=args.step_size_ms,
        model_name=args.model,
        output_path=output_path,
    )

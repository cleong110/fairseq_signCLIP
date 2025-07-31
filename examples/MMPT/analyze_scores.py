import json
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks


def plot_scores_by_query(
    df,
    x_col="window_midpoint_ms",
    y_col="score",
    query_id_col="query_id",
    label_col="query_label",
    title="Scores by Query",
    figsize=(10, 6),
    palette="tab10",
    peaks_df=None,
    height_line: float | None = None,
    max_legend_allowed=10,
):
    """
    Plot scores over time, one trace per query_id,
    using same color for shared query_label.

    Optionally, overlay peak points if `peaks_df` is provided.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if height_line is None:
        height_line = df[y_col].mean()
    ax.axhline(height_line, ls="--")

    # show_legend = df[query_id_col].nunique() <= max_legend_allowed
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=label_col if len(df[label_col].unique()) > 1 else query_id_col,
        style=query_id_col if query_id_col else None,
        estimator=None,
        palette=palette,
        # legend="full" if show_legend else False,
        legend="full",
        linewidth=1,
        alpha=0.8,
        ax=ax,
    )

    if peaks_df is not None and not peaks_df.empty:
        sns.scatterplot(
            data=peaks_df,
            x=x_col,
            y=y_col,
            hue=label_col,
            palette=palette,
            marker="X",
            s=60,
            ax=ax,
            legend=False,  # avoid duplicate legend
        )

    # Only keep entries that match the label_col values
    if query_id_col in df.columns and df[query_id_col].nunique() > max_legend_allowed:
        handles, labels = ax.get_legend_handles_labels()
        label_values = df[label_col].unique().astype(str)

        filtered = [(h, l) for h, l in zip(handles, labels) if l in label_values]

        if filtered:
            ax.legend(*zip(*filtered), title=label_col)
        else:
            ax.legend_.remove()

    # ax.set_title(title)
    x_axis_label = " ".join(xl.capitalize() for xl in x_col.split("_"))
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_col.capitalize())
    fig.tight_layout()

    return fig


def aggregate_label_by_queryid(df: pd.DataFrame, agg_fn="mean") -> pd.DataFrame:
    """
    Aggregate all query_ids together per timestep using the given aggregation function.

    Parameters:
    - df: DataFrame with columns ['window_midpoint_ms', 'score', 'query_id', 'query_label']
    - agg_fn: Aggregation function, can be 'mean', 'sum', 'max', etc.

    Returns:
    - Aggregated DataFrame with one row per x (timestep), columns ['window_midpoint_ms', 'score']
    """
    if agg_fn not in {"mean", "sum", "max", "min", "median"}:
        raise ValueError(f"Unsupported agg_fn: {agg_fn}")

    grouped = df.groupby("window_midpoint_ms")["score"]
    aggregated = getattr(grouped, agg_fn)().reset_index()
    return aggregated


def group_and_aggregate_by_label(df: pd.DataFrame, agg_fn="mean") -> pd.DataFrame:
    """
    Group by query_label, then aggregate each group using aggregate_label_by_queryid.

    Returns a combined DataFrame with an extra column 'query_label'.
    So this gives us one df each for, e.g. "CREATE", "EARTH", etc.
    """
    results = []
    for label, group in df.groupby("query_label"):
        # print(f"Aggregating {label}. Group has {len(group)}")
        agg_df = aggregate_label_by_queryid(group, agg_fn=agg_fn)
        agg_df["query_label"] = label
        results.append(agg_df)

    return pd.concat(results, ignore_index=True)


def find_peaks_in_df(
    df: pd.DataFrame,
    group_col="query_id",
    x_col="window_midpoint_ms",
    y_col="score",
    prominence=0.6,
    height=None,
) -> pd.DataFrame:
    """
    Find peaks in each group and return a DataFrame of peak positions.

    Returns:
    - DataFrame with columns [group_col, x_col, y_col]
    """
    peak_rows = []
    for query_id, group in df.groupby(group_col):
        group = group.sort_values(x_col)
        x = group[x_col].values
        y = group[y_col].values
        peaks, _ = find_peaks(y, prominence=prominence, height=height)
        for i in peaks:
            peak_rows.append(
                {
                    group_col: query_id,
                    x_col: x[i],
                    y_col: y[i],
                    "query_label": group["query_label"].iloc[0],  # to color match
                }
            )
    return pd.DataFrame(peak_rows)


def count_peaks_in_range(peaks_df: pd.DataFrame, start_ms: int, end_ms: int) -> int:
    """
    Count the number of peaks within the given time range [start_ms, end_ms].

    Parameters:
    - peaks_df: DataFrame with a column 'window_midpoint_ms' representing time of peaks.
    - start_ms: Start of the time range (inclusive).
    - end_ms: End of the time range (inclusive).

    Returns:
    - Integer count of peaks within the specified time window.
    """
    if "window_midpoint_ms" not in peaks_df.columns:
        raise ValueError("peaks_df must contain a 'window_midpoint_ms' column.")

    in_range = peaks_df[
        (peaks_df["window_midpoint_ms"] >= start_ms)
        & (peaks_df["window_midpoint_ms"] <= end_ms)
    ]
    return len(in_range)

def rank_segments(
    segments_ms_windows: list[tuple[int, int]], peaks_df, density_weight: float = 10.0
) -> list[tuple[int, tuple[int, int], float]]:
    """
    Rank segment windows by relevance based on number and density of peaks.

    Args:
        segments_ms_windows: List of (start_ms, end_ms) tuples.
        peaks_df: DataFrame containing peak timestamps.
        density_weight: Weighting factor for peak density.

    Returns:
        List of (seg_idx, (start_ms, end_ms), relevance_score) sorted by descending score.
    """
    seg_relevance_scores = []

    for seg_idx, (start_ms, end_ms) in enumerate(segments_ms_windows):
        num_peaks = count_peaks_in_range(peaks_df, start_ms=start_ms, end_ms=end_ms)

        window_length = end_ms - start_ms
        if window_length <= 0:
            print(f"Warning: Skipping invalid window ({start_ms}, {end_ms})")
            continue

        peak_density = num_peaks / window_length
        relevance_score = num_peaks + peak_density * density_weight

        print(
            f"Seg #{seg_idx}: peaks={num_peaks}, window={window_length}ms, "
            f"density={peak_density:.4f}, score={relevance_score:.4f}"
        )

        seg_relevance_scores.append((seg_idx, (start_ms, end_ms), relevance_score))

    # Sort by descending relevance score
    sorted_segments = sorted(seg_relevance_scores, key=lambda x: x[2], reverse=True)
    print(sorted_segments)
    return sorted_segments


    # Return only the windows (drop the scores)
    # return [window for window, _ in sorted_segments]




def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pose and text similarity using SignCLIP."
    )
    parser.add_argument(
        "df",
        type=Path,
        help="Path to the df",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        # default="plot.png",
        help="where to save plot",
    )
    parser.add_argument(
        "--query-json",
        type=Path,
        help="JSON with queries in it. Otherwise will look for one at pose_path.transcripts.json",
    )

    parser.add_argument(
        "--prominence",
        type=float,
        default=1.0,
        help="Passed to the find_peaks function for finding 'hits'",
    )

    parser.add_argument(
        "--height_multiplier",
        type=float,
        default=1.1,
        help="Passed to the find_peaks function for finding 'hits'",
    )

    parser.add_argument(
        "--density_weight",
        type=float,
        default=10.0,
        help="How much to weight hit density. ",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = args.df.parent

    else:
        out_dir = args.out_dir
    out_stem = args.df.stem

    transcripts = None
    if args.query_json is not None:
        with open(args.query_json) as f:
            transcripts = json.load(f)

    else:
        query_json_paths = list(args.df.parent.parent.glob("*.transcripts.json"))
        if len(query_json_paths) > 0:
            query_json_path = query_json_paths[0]
            print(f"Loading JSON from {query_json_path}")
            with open(query_json_path) as f:
                transcripts = json.load(f)

    with open(args.df.parent.parent / "pose_info.json") as f:
        pose_info = json.load(f)
    print(pose_info)

    segments_ms_windows = []
    if transcripts is not None:
        for seg_idx, transcript in enumerate(transcripts):
            # print(transcript)
            start_frame = transcript["start_frame"]
            end_frame = transcript["end_frame"]

            start_ms = 1000 * start_frame / pose_info["fps"]
            end_ms = 1000 * end_frame / pose_info["fps"]

            print(
                f"seg {seg_idx} goes from {start_frame} ({start_ms} ms) to {end_frame} ({end_ms} ms)"
            )
            window = (start_ms, end_ms)
            segments_ms_windows.append(window)

    df = pd.read_parquet(args.df)

    # Optionally compute and overlay peaks
    height = df["score"].mean() * args.height_multiplier
    peaks_df = find_peaks_in_df(df, prominence=args.prominence, height=height)
    fig = plot_scores_by_query(df, peaks_df=peaks_df, height_line=height)

    plot_out = out_dir / f"{out_stem}.png"
    fig.savefig(plot_out)
    print(plot_out.resolve())

    # Also plot for all scores for each query_label
    for query_label, query_df in df.groupby("query_label"):
        height = query_df["score"].mean() * args.height_multiplier
        query_peaks_df = find_peaks_in_df(
            query_df, prominence=args.prominence, height=height
        )
        # fig = plot_scores_by_query(
        #     query_df, peaks_df=query_peaks_df, height_line=height
        # )
        fig = plot_scores_by_query(
            query_df, height_line=height
        )
        plot_out = out_dir / f"{out_stem}_{query_label}_notaggregated.png"
        fig.savefig(plot_out)
        print(plot_out.resolve())

    for agg_fn in ["mean", "sum", "min"]:
        agg_df = group_and_aggregate_by_label(df, agg_fn=agg_fn)
        # print(agg_df.head())
        mean_score = agg_df["score"].mean()
        height = mean_score * args.height_multiplier
        agg_peaks_df = find_peaks_in_df(
            agg_df,
            group_col="query_label",
            prominence=args.prominence,
            height=height,
        )
        print(agg_peaks_df.head())
        fig = plot_scores_by_query(
            agg_df,
            x_col="window_midpoint_ms",
            y_col="score",
            query_id_col=None,  # no line-style distinctions needed
            label_col="query_label",
            title=f"{agg_fn.capitalize()} Aggregated Scores by Label",
            peaks_df=agg_peaks_df,
            height_line=height,
        )
        agg_out = out_dir / f"{out_stem}_{agg_fn}.png"
        fig.savefig(agg_out)
        print(agg_out.resolve())
        rank_segments(segments_ms_windows, agg_peaks_df, args.density_weight)

    agg_df = group_and_aggregate_by_label(df, agg_fn="mean")
    print(agg_df.head())

    for query_label, query_label_df in agg_df.groupby("query_label"):
        mean_score = query_label_df["score"].mean()
        height = mean_score * args.height_multiplier
        agg_peaks_df = find_peaks_in_df(
            query_label_df,
            group_col="query_label",
            prominence=1,
            height=height,
        )
        fig = plot_scores_by_query(
            query_label_df,
            x_col="window_midpoint_ms",
            y_col="score",
            query_id_col=None,  # no line-style distinctions needed
            label_col="query_label",
            title=f"{query_label} Aggregated Scores by Label",
            peaks_df=agg_peaks_df,
            height_line=height,
        )
        query_label_out = out_dir / f"{out_stem}_mean_{query_label}.png"
        fig.savefig(query_label_out)
        print(query_label_out.resolve())

    # get the segments


if __name__ == "__main__":
    main()

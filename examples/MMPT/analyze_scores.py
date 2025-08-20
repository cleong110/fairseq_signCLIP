from collections import defaultdict
import json
from pathlib import Path
import argparse
import pandas as pd
from scipy.signal import find_peaks
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469

# def plot_scores_by_query(
#     df,
#     x_col="window_midpoint_ms",
#     y_col="score",
#     query_id_col="query_id",
#     label_col="query_label",
#     title="Scores by Query",
#     figsize=(10, 6),
#     palette="tab10",
#     peaks_df=None,
#     height_line: float | None = None,
#     max_legend_allowed=10,
# ):
#     """
#     Plot scores over time, one trace per query_id,
#     using same color for shared query_label.

#     Optionally, overlay peak points if `peaks_df` is provided.
#     """
#     fig, ax = plt.subplots(figsize=figsize)

#     if height_line is None:
#         height_line = df[y_col].mean()
#     ax.axhline(height_line, ls="--")

#     # show_legend = df[query_id_col].nunique() <= max_legend_allowed
#     sns.lineplot(
#         data=df,
#         x=x_col,
#         y=y_col,
#         hue=label_col if len(df[label_col].unique()) > 1 else query_id_col,
#         style=query_id_col if query_id_col else None,
#         estimator=None,
#         palette=palette,
#         # legend="full" if show_legend else False,
#         legend="full",
#         linewidth=1,
#         alpha=0.8,
#         ax=ax,
#     )

#     if peaks_df is not None and not peaks_df.empty:
#         sns.scatterplot(
#             data=peaks_df,
#             x=x_col,
#             y=y_col,
#             hue=label_col,
#             palette=palette,
#             marker="X",
#             s=60,
#             ax=ax,
#             legend=False,  # avoid duplicate legend
#         )

#     # Only keep entries that match the label_col values
#     if query_id_col in df.columns and df[query_id_col].nunique() > max_legend_allowed:
#         handles, labels = ax.get_legend_handles_labels()
#         label_values = df[label_col].unique().astype(str)

#         filtered = [(h, l) for h, l in zip(handles, labels) if l in label_values]

#         if filtered:
#             ax.legend(*zip(*filtered), title=label_col)
#         else:
#             ax.legend_.remove()

#     # ax.set_title(title)
#     x_axis_label = " ".join(xl.capitalize() for xl in x_col.split("_"))
#     ax.set_xlabel(x_axis_label)
#     ax.set_ylabel(y_col.capitalize())
#     fig.tight_layout()

#     return fig


def plot_scores_by_query(
    df: pd.DataFrame,
    x_col: str = "window_midpoint_ms",
    y_col: str = "score",
    query_id_col: str = "query_id",
    label_col: str = "query_label",
    title: str = "Scores by Query",
    palette: str | None = None,  # Plotly handles colors automatically
    peaks_df: pd.DataFrame | None = None,
    height_line: float | None = None,
    max_legend_allowed: int = 10,
):
    """
    Interactive Plotly version of scores plot:
    - One line per query_id
    - Consistent colors across shared query_label
    - Optional peak markers
    - Horizontal threshold line
    - Legend filtering
    """
    # Default threshold line
    if height_line is None:
        height_line = df[y_col].mean()

    # Plotly line figure
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=label_col if df[label_col].nunique() > 1 else query_id_col,
        line_dash=query_id_col if query_id_col else None,
        title=title,
    )

    # Add horizontal reference line
    fig.add_hline(
        y=height_line,
        line_dash="dash",
        line_color="gray",
        annotation_text="Threshold",
        annotation_position="top left",
    )

    # Add peak markers if provided
    if peaks_df is not None and not peaks_df.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks_df[x_col],
                y=peaks_df[y_col],
                mode="markers",
                marker=dict(symbol="x", size=10, color="black"),
                name="Peaks",
                showlegend=False,
            )
        )

    # Axis labels
    x_axis_label = " ".join(xl.capitalize() for xl in x_col.split("_"))
    fig.update_layout(
        xaxis_title=x_axis_label,
        yaxis_title=y_col.capitalize(),
        legend_title=label_col,
    )

    # Optionally trim legend if too many query_ids
    if query_id_col in df.columns:
        if df[query_id_col].nunique() > max_legend_allowed:
            # Keep only label_col values
            valid_labels = set(df[label_col].astype(str).unique())
            fig.for_each_trace(
                lambda trace: trace.update(showlegend=trace.name in valid_labels)
            )

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

    return results


def find_peaks_in_df(
    df: pd.DataFrame,
    group_col="query_id",
    x_col="window_midpoint_ms",
    y_col="score",
    prominence=0.6,
    height=None,
    width=None,  # in number of samples. TODO: set based on step size?
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

        # TODO: box plots of properties to investigate
        peaks, properties = find_peaks(
            y, prominence=prominence, height=height, width=width
        )
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
) -> pd.DataFrame:
    """
    Rank segment windows by relevance based on number and density of peaks.

    Args:
        segments_ms_windows: List of (start_ms, end_ms) tuples.
        peaks_df: DataFrame containing peak timestamps.
        density_weight: Weighting factor for peak density.

    Returns:
        A DataFrame sorted by descending relevance score, with columns:
        ['seg_idx', 'start_ms', 'end_ms', 'num_peaks', 'density', 'relevance_score', 'rank']
    """
    seg_relevance_data = defaultdict(list)

    print(peaks_df.columns)

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

        seg_relevance_data["seg_idx"].append(seg_idx)
        seg_relevance_data["start_ms"].append(start_ms)
        seg_relevance_data["end_ms"].append(end_ms)
        seg_relevance_data["num_peaks"].append(num_peaks)
        seg_relevance_data["density"].append(peak_density)
        seg_relevance_data["relevance_score"].append(relevance_score)

    df = pd.DataFrame(seg_relevance_data)

    if df.empty:
        print("Warning: No valid segments found.")
        return df

    df = df.sort_values(by="relevance_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1  # Rank starting from 1

    return df


def full_df_analysis(
    df_path: Path,
    out_dir: Path | None = None,
    query_json: Path | None = None,
    prominence: float = 1.0,
    height_multiplier: float = 1.1,
    density_weight: float = 10.0,
):
    if out_dir is None:
        out_dir = df_path.parent
    out_stem = df_path.stem

    transcripts = None
    if query_json is not None:
        query_json_path = query_json
        with open(query_json_path) as f:
            transcripts = json.load(f)
    else:
        query_json_paths = list(df_path.parent.parent.glob("*.transcripts.json"))
        if len(query_json_paths) > 0:
            query_json_path = query_json_paths[0]
            print(f"Loading JSON from {query_json_path}")
            with open(query_json_path) as f:
                transcripts = json.load(f)
        else:
            query_json_path = None

    with open(df_path.parent.parent / "pose_info.json") as f:
        pose_info = json.load(f)
    print(pose_info)

    segments_ms_windows = []
    true_seg_idx = None
    original_query = None
    if transcripts is not None:
        for seg_idx, transcript in enumerate(transcripts):
            if df_path.parent.name == f"seg_idx{seg_idx}":
                true_seg_idx = seg_idx
                original_query = transcript["text"]
                print(f"TRUE SEGMENT INDEX: {true_seg_idx}")

            start_frame = transcript["start_frame"]
            end_frame = transcript["end_frame"]

            start_ms = 1000 * start_frame / pose_info["fps"]
            end_ms = 1000 * end_frame / pose_info["fps"]

            print(
                f"seg {seg_idx} goes from {start_frame} ({start_ms} ms) to {end_frame} ({end_ms} ms)"
            )
            window = (start_ms, end_ms)
            segments_ms_windows.append(window)

    df = pd.read_parquet(df_path)
    df["time_hms"] = pd.to_timedelta(df["window_midpoint_ms"], unit="ms")

    agg_by_label_dfs = group_and_aggregate_by_label(df, agg_fn="mean")

    all_mean_df = pd.concat(agg_by_label_dfs, ignore_index=True)

    agg_by_label_dfs.append(all_mean_df)
    for agg_df in agg_by_label_dfs:
        labels = agg_df["query_label"].unique()
        # scores_with_these_labels_df = df[df["query_label"].isin(labels)]
        labels_joined = "_".join(labels)
        print(agg_df.columns)

        height = agg_df["score"].mean() * height_multiplier
        max_val = agg_df["score"].max()
        print(f"Height for {labels}: {height}. Max: {max_val}")

        peaks_for_mean_df = find_peaks_in_df(
            agg_df, prominence=prominence, height=height, group_col="query_label"
        )
        print(peaks_for_mean_df)
        print(f"Found {len(peaks_for_mean_df)} peaks for {labels}")

        # print(scores_with_these_labels_df.columns)

        fig = plot_scores_by_query(
            agg_df,
            x_col="window_midpoint_ms",
            y_col="score",
            query_id_col=None,
            label_col="query_label",
            title="Mean Aggregated Scores by Query Label",
            peaks_df=peaks_for_mean_df,
            height_line=height,
        )

        mean_out_html = out_dir / f"{out_stem}_{labels_joined}_mean.html"
        mean_out_pdf = out_dir / f"{out_stem}_{labels_joined}_mean.pdf"

        # Save interactive HTML
        fig.write_html(str(mean_out_html))

        fig.write_image(str(mean_out_pdf))
        print(mean_out_pdf.resolve())
        if "GOD" == labels[0] and len(labels) == 1:
            exit()

    segment_rankings_df = rank_segments(
        segments_ms_windows, peaks_for_mean_df, density_weight
    )

    if true_seg_idx is not None and original_query is not None:
        print(true_seg_idx, original_query)
        video_id = query_json_path.name.split(".")[0]
        segment_rankings_df["video_id"] = video_id
        segment_rankings_df["query_text"] = original_query

    desired_columns = [
        "video_id",
        "query_text",
        "rank",
        "seg_idx",
        "start_ms",
        "end_ms",
        "num_peaks",
        "density",
        "relevance_score",
    ]
    existing_columns = [
        col for col in desired_columns if col in segment_rankings_df.columns
    ]
    remaining_columns = [
        col for col in segment_rankings_df.columns if col not in existing_columns
    ]
    segment_rankings_df = segment_rankings_df[existing_columns + remaining_columns]
    segment_rankings_df["path_to_results"] = df_path

    print(segment_rankings_df)
    predictions_out = out_dir / "segment_rankings.csv"
    segment_rankings_df.to_csv(predictions_out, index=False)
    print(predictions_out.resolve())
    return segment_rankings_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pose and text similarity using SignCLIP."
    )
    parser.add_argument("df", type=Path, help="Path to the df")
    parser.add_argument("--out-dir", type=Path, help="Where to save plot")
    parser.add_argument("--query-json", type=Path, help="JSON with queries in it")
    parser.add_argument("--prominence", type=float, default=1.0, help="Find peaks")
    parser.add_argument(
        "--height_multiplier", type=float, default=1.1, help="Peak height multiplier"
    )
    parser.add_argument(
        "--density_weight", type=float, default=10.0, help="Weight hit density"
    )

    args = parser.parse_args()

    if args.df.is_dir():
        df_paths = args.df.rglob("*all_scores.parquet")
        ground_truth_files = list(args.df.glob("*.transcripts.json"))
        assert len(ground_truth_files) == 1

        # create ground truth CSV at top level
        # query_text,video_id,seg_idx,start_frame,end_frame,total_frames
        with open(ground_truth_files[0]) as f:
            ground_truth_transcripts = json.load(f)
            ground_truth_dict = defaultdict(list)
            for seg_idx, transcript in enumerate(ground_truth_transcripts):
                ground_truth_dict["query_text"].append(transcript["text"])
                ground_truth_dict["video_id"].append(
                    ground_truth_files[0].name.split(".")[0]
                )
                ground_truth_dict["seg_idx"].append(seg_idx)
                ground_truth_dict["start_frame"].append(transcript["start_frame"])
                ground_truth_dict["end_frame"].append(transcript["end_frame"])
            ground_truth_df = pd.DataFrame(ground_truth_dict)
            out = args.df / "ground_truth.csv"
            ground_truth_df.to_csv(out, index=False)
            print(f"Saved ground truth to {out.resolve()}")

    else:
        df_paths = [args.df]

    all_segment_ranking_dfs_list = []
    for df_path in df_paths:
        segment_rankings_df = full_df_analysis(
            df_path=df_path,
            out_dir=args.out_dir,
            query_json=args.query_json,
            prominence=args.prominence,
            height_multiplier=args.height_multiplier,
            density_weight=args.density_weight,
        )
        all_segment_ranking_dfs_list.append(segment_rankings_df)
    all_segment_rankings_df = pd.concat(all_segment_ranking_dfs_list)

    # Drop rows where 'video_id' or 'query_text' is NaN or empty string
    all_segment_rankings_df = all_segment_rankings_df.dropna(
        subset=["video_id", "query_text"]
    )
    all_segment_rankings_df = all_segment_rankings_df[
        (all_segment_rankings_df["video_id"].str.strip() != "")
        & (all_segment_rankings_df["query_text"].str.strip() != "")
    ]

    if args.df.is_dir():
        out = args.df / "all_predictions.csv"
        all_segment_rankings_df.to_csv(out, index=False)
        print(out.resolve())


if __name__ == "__main__":
    main()

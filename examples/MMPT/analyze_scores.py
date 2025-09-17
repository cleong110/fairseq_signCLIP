import json
from pathlib import Path
import argparse
import pandas as pd
from scipy.signal import find_peaks
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def plot_scores_by_query(
    df: pd.DataFrame,
    x_col: str = "window_midpoint_ms",
    y_col: str = "score",
    query_id_col: str = "query_id",
    label_col: str = "query_label",
    title: str | None = None,
    palette: list[str] | None = None,
    peaks_df: pd.DataFrame | None = None,
    height_line: float | None = None,
    max_legend_allowed: int = 10,
):
    """
    Interactive Plotly version of scores plot:
    - One line per query_id
    - Consistent colors across shared query_label
    - Optional peak markers colored by label
    - Horizontal threshold as y-axis tick
    - Legend filtering
    - X-axis shows mm:ss using datetime64[ns] anchored at zero
    """
    df = df.copy()
    df["x_time"] = pd.to_datetime(df[x_col], unit="ms")

    if peaks_df is not None and not peaks_df.empty:
        peaks_df = peaks_df.copy()
        peaks_df["x_time"] = pd.to_datetime(peaks_df[x_col], unit="ms")

    # Default threshold line
    if height_line is None:
        height_line = df[y_col].mean()

    color_by = label_col if df[label_col].nunique() > 1 else query_id_col

    # Generate line plot
    fig = px.line(
        df,
        x="x_time",
        y=y_col,
        color=color_by,
        line_dash=query_id_col if query_id_col else None,
        title=title,
        color_discrete_sequence=palette or px.colors.qualitative.Plotly,
    )

    # Extract color mapping from the line traces
    label_to_color = {}
    for trace in fig.data:
        if hasattr(trace, "legendgroup"):
            label_to_color[trace.legendgroup] = trace.line.color

    # Add peaks with matching colors
    if peaks_df is not None and not peaks_df.empty:
        for label, group in peaks_df.groupby(label_col):
            fig.add_trace(
                go.Scatter(
                    x=group["x_time"],
                    y=group[y_col],
                    mode="markers",
                    marker=dict(
                        symbol="x", size=10, color=label_to_color.get(label, "black")
                    ),
                    name=f"Peaks: {label}",
                    showlegend=False,
                )
            )

    # Add horizontal threshold line
    # how to add hline with annotation: https://community.plotly.com/t/how-to-add-a-single-tick-value-to-default-ticks-on-an-axis/45621/3
    fig.add_hline(
        y=height_line,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Threshold<br>{height_line:.2f}",
        annotation_position="right",
    )

    # Axis formatting
    existing_yticks = list(fig.layout.yaxis.tickvals or [])
    if height_line not in existing_yticks:
        existing_yticks.append(height_line)
    existing_yticks = sorted(existing_yticks)

    yaxis_title = " ".join(s.capitalize() for s in y_col.split("_"))
    legend_title = " ".join(s.capitalize() for s in label_col.split("_"))

    # Update layout
    fig.update_layout(
        xaxis_title="Time (mm:ss)",
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        xaxis=dict(tickformat="%M:%S"),
        yaxis=dict(
            # tickvals=existing_yticks,
            # ticktext=[
            #     f"{v:.1f}" if v != height_line else f"{v:.1f} (threshold)"
            #     for v in existing_yticks
            # ],
        ),
    )

    # Trim legend if too many labels
    if query_id_col in df.columns and df[query_id_col].nunique() > max_legend_allowed:
        valid_labels = set(df[label_col].astype(str).unique())
        fig.for_each_trace(
            lambda trace: trace.update(showlegend=trace.name in valid_labels)
        )

    # put the legend on top

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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


def full_df_analysis(
    df_path: Path,
    out_dir: Path | None = None,
    query_json: Path | None = None,
    prominence: float = 1.0,
    height_multiplier: float = 1.1,
    density_weight: float = 10.0,
    filter_eng: bool = False,
):
    if out_dir is None:
        out_dir = df_path.parent
    out_stem = df_path.stem

    print("Analyzing {df_path}")

    with open(df_path.parent.parent / "pose_info.json") as f:
        pose_info = json.load(f)
    print(pose_info)

    df = pd.read_parquet(df_path)
    print(df["query_label"].unique())
    df["time_hms"] = pd.to_timedelta(df["window_midpoint_ms"], unit="ms")

    if filter_eng:
        df = df[~df["query_id"].str.contains("(eng)", regex=False)]
        if len(df) == 0:
            raise ValueError(
                f"Filtering out English queries leaves our scores empty - df len(): {len(df)}"
            )

    agg_by_label_dfs = group_and_aggregate_by_label(df, agg_fn="mean")

    all_mean_df = pd.concat(agg_by_label_dfs, ignore_index=True)

    subset_custom_looking_deletethis = [
        "GOD",
        "KNOW",
        "TREE",
        "WIFE",
        "HEAD",
        "PAIN",
        "MAN",
        "DAY",
        "HEAVEN",
        "EARTH",
    ]

    if any(
        l in all_mean_df["query_label"].unique().tolist()
        # for l in ["GOD", "HEAVEN", "EARTH", "DAY"]
        for l in subset_custom_looking_deletethis
    ):
        print("found it")
        custom_mean_df = all_mean_df[
            all_mean_df["query_label"].isin(subset_custom_looking_deletethis)
        ]
        print(custom_mean_df)
        agg_by_label_dfs.append(custom_mean_df)

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
            title=None,  # "Mean Aggregated Scores by Query Label",
            peaks_df=peaks_for_mean_df,
            height_line=height,
        )

        settings_string = f"prominence_{prominence}_heightmult_{height_multiplier}"
        if filter_eng:
            settings_string = f"{settings_string}_NoEnglish"

        labels_joined_out_dir = out_dir / "peaks" / settings_string / f"{labels_joined}"
        labels_joined_out_dir.mkdir(parents=True, exist_ok=True)

        mean_out_html = labels_joined_out_dir / f"{out_stem}_{labels_joined}_mean.html"
        mean_out_pdf = labels_joined_out_dir / f"{out_stem}_{labels_joined}_mean.pdf"
        mean_out_png = labels_joined_out_dir / f"{out_stem}_{labels_joined}_mean.png"

        peaks_for_mean_df.to_csv(
            labels_joined_out_dir / f"{out_stem}_{labels_joined}_peaks.csv", index=False
        )

        # Save interactive HTML
        fig.write_html(str(mean_out_html))

        fig.write_image(str(mean_out_pdf))
        fig.write_image(str(mean_out_png))
        print(mean_out_pdf.resolve())
        print(mean_out_png.resolve())


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
    parser.add_argument(
        "--filter-eng",
        action="store_true",
        help="whether to filter out the (eng) queries before finding peaks",
    )

    args = parser.parse_args()

    if args.df.is_dir():
        df_paths = list(args.df.rglob("*all_scores.parquet"))
        # ground_truth_files = list(args.df.glob("*.transcripts.json"))
        # assert len(ground_truth_files) == 1, f"{args.df} has no ground truth transcripts!"

        # # create ground truth CSV at top level
        # # query_text,video_id,seg_idx,start_frame,end_frame,total_frames
        # with open(ground_truth_files[0]) as f:
        #     ground_truth_transcripts = json.load(f)
        #     ground_truth_dict = defaultdict(list)
        #     for seg_idx, transcript in enumerate(ground_truth_transcripts):
        #         ground_truth_dict["query_text"].append(transcript["text"])
        #         ground_truth_dict["video_id"].append(
        #             ground_truth_files[0].name.split(".")[0]
        #         )
        #         ground_truth_dict["seg_idx"].append(seg_idx)
        #         ground_truth_dict["start_frame"].append(transcript["start_frame"])
        #         ground_truth_dict["end_frame"].append(transcript["end_frame"])
        #     ground_truth_df = pd.DataFrame(ground_truth_dict)
        #     out = args.df / "ground_truth.csv"
        #     ground_truth_df.to_csv(out, index=False)
        #     print(f"Saved ground truth to {out.resolve()}")

    else:
        df_paths = [args.df]

    for df_path in df_paths:
        if "seg_idx9999" not in df_path.parent.name:
            # print(f"Skipping {df_path}")
            continue
        full_df_analysis(
            df_path=df_path,
            out_dir=args.out_dir,
            query_json=args.query_json,
            prominence=args.prominence,
            height_multiplier=args.height_multiplier,
            density_weight=args.density_weight,
            filter_eng=args.filter_eng,
        )


if __name__ == "__main__":
    main()

# Usage:
# python analyze_scores.py "/opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/results/asl_finetune_checkpoint_best/samplespergloss_15/start_0_end_None/windowsize1000_step200/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-001-ase-3-passage _ god creates the world.pose-mediapipe/" --height_multiplier 1.2
# OK let's say you previously did find /data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ -name "*passage _ *first*man*and*woman*disobey*.pose-mediapipe*.pose"|parallel -j1 python /opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/search_with_text_and_poses.py --pose_path "{}" --start_time_ms 0 --step_size_ms 200 --window_size_ms 1000 --eng "refrigerator" --model "asl_finetune_checkpoint_best" --samples-per-gloss 15
# then you might do:
# python analyze_scores.py "/opt/home/cleong/projects/semantic-sign-language-search/setup_signCLIP/fairseq/examples/MMPT/results/asl_finetune_checkpoint_best/samplespergloss_15/start_0_end_None/windowsize1000_step200/ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt-003-ase-3-passage _ the first man and woman disobey god.pose-mediapipe" --height_multiplier 1.2

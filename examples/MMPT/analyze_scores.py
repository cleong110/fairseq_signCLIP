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
    plot_avg_y=True,
):
    """
    Plot scores over time, one trace per query_id,
    using same color for shared query_label.

    Optionally, overlay peak points if `peaks_df` is provided.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if plot_avg_y:
        avg_y = df[y_col].mean()
        ax.axhline(avg_y, ls='--')

    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=label_col,
        style=query_id_col if query_id_col else None,
        estimator=None,
        palette=palette,
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

    ax.set_title(title)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    fig.tight_layout()

    return fig


def aggregate_label(df: pd.DataFrame, agg_fn="mean") -> pd.DataFrame:
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
    Group by query_label, then aggregate each group using aggregate_label.

    Returns a combined DataFrame with an extra column 'query_label'.
    """
    results = []
    for label, group in df.groupby("query_label"):
        agg_df = aggregate_label(group, agg_fn=agg_fn)
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
    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = args.df.parent

    else:
        out_dir = args.out_dir
    out_stem = args.df.stem

    df = pd.read_parquet(args.df)
    # Optionally compute and overlay peaks
    peaks_df = find_peaks_in_df(df, prominence=0.6)
    fig = plot_scores_by_query(df, peaks_df=peaks_df)

    plot_out = out_dir / f"{out_stem}.png"
    fig.savefig(plot_out)
    print(plot_out.resolve())

    for agg_fn in ["mean", "sum", "min"]:
        agg_df = group_and_aggregate_by_label(df, agg_fn=agg_fn)
        print(agg_df.head())
        mean_score = agg_df["score"].mean()
        agg_peaks_df = find_peaks_in_df(agg_df, group_col="query_label", prominence=1, height=mean_score)
        fig = plot_scores_by_query(
            agg_df,
            x_col="window_midpoint_ms",
            y_col="score",
            query_id_col=None,  # no line-style distinctions needed
            label_col="query_label",
            title=f"{agg_fn.capitalize()} Aggregated Scores by Label",
            peaks_df=agg_peaks_df,
        )
        agg_out = out_dir / f"{out_stem}_{agg_fn}.png"
        fig.savefig(agg_out)
        print(agg_out.resolve())


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scores_by_query(
    df,
    x_col="window_midpoint_ms",  # or time/index
    y_col="score",
    query_id_col="query_id",
    label_col="query_label",
    title="Scores by Query",
    figsize=(10, 6),
    palette="tab10",
):
    """
    Plot scores over time/frames, one trace per query_id,
    using same color for shared query_label.

    Parameters:
    - df: pd.DataFrame with columns [x_col, y_col, query_id_col, label_col]
    - x_col: column to use for x-axis (e.g., frame/time)
    - y_col: column with the score to plot
    - query_id_col: unique identifier for each query (each gets a trace)
    - label_col: label that defines the color grouping
    - title: plot title
    - figsize: size of the plot
    - palette: seaborn color palette or dict mapping labels to colors
    """
    fig, ax = plt.subplots(figsize=figsize)

    print(df.columns)

    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=label_col,
        style=query_id_col,
        estimator=None,
        palette=palette,
        legend="full",
        linewidth=1,
        alpha=0.8,
        ax=ax  # <--- plot into this specific Axes
    )

    ax.set_title(title)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    fig.tight_layout()

    return fig


def aggregate_label(df: pd.DataFrame, agg_fn='mean') -> pd.DataFrame:
    """
    Aggregate all query_ids together per timestep using the given aggregation function.

    Parameters:
    - df: DataFrame with columns ['window_midpoint_ms', 'score', 'query_id', 'query_label']
    - agg_fn: Aggregation function, can be 'mean', 'sum', 'max', etc.

    Returns:
    - Aggregated DataFrame with one row per x (timestep), columns ['window_midpoint_ms', 'score']
    """
    if agg_fn not in {'mean', 'sum', 'max', 'min', 'median'}:
        raise ValueError(f"Unsupported agg_fn: {agg_fn}")

    grouped = df.groupby("window_midpoint_ms")["score"]
    aggregated = getattr(grouped, agg_fn)().reset_index()
    return aggregated


def group_and_aggregate_by_label(df: pd.DataFrame, agg_fn='mean') -> pd.DataFrame:
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
    fig = plot_scores_by_query(df)

    plot_out = out_dir/f"{out_stem}.png"
    fig.savefig(plot_out)
    print(plot_out.resolve())

    for agg_fn in ["mean", "sum", "min"]:
        agg_df = group_and_aggregate_by_label(df, agg_fn=agg_fn)
        fig = plot_scores_by_query(
            agg_df,
            x_col="window_midpoint_ms",
            y_col="score",
            query_id_col=None,  # no line-style distinctions needed
            label_col="query_label",
            title=f"{agg_fn.capitalize()} Aggregated Scores by Label",
        )
        agg_out = out_dir/f"{out_stem}_{agg_fn}.png"
        fig.savefig(agg_out)
        print(agg_out.resolve())


if __name__ == "__main__":
    main()

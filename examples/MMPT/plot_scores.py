from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import plotly.graph_objects as go
# import pandas as pd
# from plotly.colors import qualitative

# def plotly_scores_by_query(
#     df: pd.DataFrame,
#     x_col: str = "window_midpoint_ms",
#     y_col: str = "score",
#     query_id_col: str = "query_id",
#     label_col: str = "query_label",
#     title: str = "Scores by Query",
#     color_palette=None,
# ) -> go.Figure:
#     """
#     Plot an interactive Plotly line plot of scores per query_id,
#     with shared color for identical query_labels.

#     Parameters:
#     - df: DataFrame containing the data
#     - x_col: column for x-axis (e.g., frame, time)
#     - y_col: column for y-axis (score)
#     - query_id_col: unique identifier per trace
#     - label_col: shared label to group colors
#     - title: plot title
#     - color_palette: optional dict[label] -> color or list of colors

#     Returns:
#     - Plotly Figure object
#     """
#     fig = go.Figure()

#     # Build color map: one color per query_label
#     unique_labels = df[label_col].unique()
#     if color_palette is None:
#         base_colors = qualitative.Set2 + qualitative.Set3 + qualitative.Plotly
#         color_palette = {
#             label: base_colors[i % len(base_colors)]
#             for i, label in enumerate(unique_labels)
#         }

#     # Plot each query_id separately
#     for query_id, subdf in df.groupby(query_id_col):
#         label = subdf[label_col].iloc[0]
#         fig.add_trace(go.Scatter(
#             x=subdf[x_col],
#             y=subdf[y_col],
#             mode="lines",
#             name=f"{label} / {query_id}",
#             line=dict(color=color_palette[label]),
#             hovertemplate=f"{label}<br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra>{query_id}</extra>",
#         ))

#     fig.update_layout(
#         title=title,
#         xaxis_title=x_col.capitalize(),
#         yaxis_title=y_col.capitalize(),
#         legend_title="Query Label / ID",
#         template="plotly_white",
#         height=600,
#     )

#     return fig


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
        "--out",
        type=Path,
        default="plot.png",
        help="where to save plot",
    )    
    args = parser.parse_args()

    df = pd.read_parquet(args.df)
    fig = plot_scores_by_query(df)
    fig.savefig(args.out)
    print(args.out.resolve())

if __name__ == "__main__":
    main()

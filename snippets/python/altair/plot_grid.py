"""
TAGS: altair|chart|datavis|dataviz|data vis|data viz|graph|grid|plot|plot grid|subplot|subplots|visualisation|visualization
DESCRIPTION: Convert multiple altair plots into a single plot by laying them out in a grid 
REQUIREMENTS: pip install altair
"""

import altair as alt

alt.renderers.enable("browser")  # makes .display() render plot in a new browser window

def altair_plotgrid(
    plots_list: list[alt.Chart|alt.LayerChart],
    n_plots_per_row: int,
) -> alt.VConcatChart:
    """
    Combines a list of altair plots into a single plot by laying them out in a grid
    """
    plot_rows: list[list[alt.Chart|alt.LayerChart]] = []
    for plot_num, plot in enumerate(plots_list):
        plot_row_num: int = plot_num // n_plots_per_row
        try:
            plot_rows[plot_row_num].append(plot)
        except IndexError:
            plot_rows.append([plot])
    return alt.vconcat(*[alt.hconcat(*plot_row) for plot_row in plot_rows])

data = [
    {"x": 1, "y": 2},
    {"x": 2, "y": 3},
    {"x": 3, "y": 5},
    {"x": 4, "y": 4},
    {"x": 5, "y": 6},
]

scatter_plot = (
    alt.Chart(alt.Data(values=data))
    .mark_circle(size=60)
    .encode(
        x=alt.X("x:Q", title="X Axis Label"),
        y=alt.Y("y:Q", title="Y Axis Label"),
        tooltip=["x:Q", "y:Q"],  # (optional) adds point info on hover
    )
    .properties(title="Basic Scatter Plot")
)

altair_plotgrid(
    plots_list = [scatter_plot, scatter_plot, scatter_plot],
    n_plots_per_row=2,
).display()
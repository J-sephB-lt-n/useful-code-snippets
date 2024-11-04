"""
TAGS: altair|chart|datavis|dataviz|data vis|data viz|graph|plot|scatter|scatterplot|visualisation|visualization
DESCRIPTION: Draw a basic scatterplot using the `altair` python library
REQUIREMENTS: pip install altair
"""

import altair as alt

alt.renderers.enable("browser")  # makes .display() render plot in a new browser window

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
scatter_plot.display()

# colouring points by group #
data = [
    {"x": 1, "y": 2, "group": "police"},
    {"x": 2, "y": 3, "group": "police"},
    {"x": 3, "y": 5, "group": "bad guys"},
    {"x": 4, "y": 4, "group": "bad guys"},
    {"x": 5, "y": 6, "group": "police"},
]
scatter_plot = (
    alt.Chart(alt.Data(values=data))
    .mark_circle(size=60)
    .encode(
        x=alt.X("x:Q", title="X Axis Label"),
        y=alt.Y("y:Q", title="Y Axis Label"),
        color=alt.Color("group:N", title="Group"),
        tooltip=["x:Q", "y:Q", "group:N"],  # (optional) adds point info on hover
    )
    .properties(title="Basic Scatter Plot with Grouping")
)
scatter_plot.display()

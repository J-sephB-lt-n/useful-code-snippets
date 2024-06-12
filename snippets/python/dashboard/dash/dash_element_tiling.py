"""
TAGS: bootstrap|bootstrap components|col|column|columns|components|dash|dashboard|data visualisation|data visualization|datavis|dataviz|frontend|grid|gui|html|layout|plotly|row|rows|ui|user interface|visualisation|visualization|web|website
DESCRIPTION: An example of how to lay out elements in a grid in a Dash dashboard using dash-bootstrap-components 
REQUIREMENTS: pip install "dash==2.17.0" "dash-bootstrap-components==1.6.0"
USAGE: $ python dash_element_tiling.py
"""

import dash_bootstrap_components as dbc
from dash import Dash, html

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

BOX_STYLE = {"background-color": "#ADD8E6", "text-align": "center"}

"""
Global layout is: 

+------+------+------+
| Box1 | Box2 | Box3 |
+------+------+------+
| Box4 | Box5 | Box6 |
+------+------+------+
"""


box2 = dbc.Col("2", style=BOX_STYLE, width=2)
box3 = dbc.Stack(
    [
        dbc.Col("3A", style=BOX_STYLE, width=5),
        dbc.Col("3B", style=BOX_STYLE, width=5),
    ],
    direction="vertical",
    gap=1,
    className="justify-content-center align-items-center",
)
box4 = dbc.Col(
    "4",
    style=BOX_STYLE,
    width=5,
    className="justify-content-center align-items-center",
)
box5 = dbc.Col(
    "5",
    style=BOX_STYLE,
    width=2,
    className="justify-content-center align-items-center",
)
box6 = dbc.Col(
    "6",
    style=BOX_STYLE,
    width=5,
    className="justify-content-center align-items-center",
)
box1 = dbc.Stack(
    # Box 1 layout looks like this:
    # +---------+---------+
    # |    a    |    b    |
    # +---------+---------+
    # | c | d | e | f | g |
    # +---------+---------+
    # |   h     |    i    |
    # |   j     |    k    |
    # +-------------------+
    # | l  | m  | n  | o  |
    # +-------------------+
    [
        dbc.Stack(
            [
                dbc.Col("a", style=BOX_STYLE),
                dbc.Col("b", style=BOX_STYLE),
            ],
            direction="horizontal",
            gap=1,
        ),
        dbc.Stack(
            [
                dbc.Col("c", style=BOX_STYLE),
                dbc.Col("d", style=BOX_STYLE),
                dbc.Col("e", style=BOX_STYLE),
                dbc.Col("f", style=BOX_STYLE),
                dbc.Col("g", style=BOX_STYLE),
            ],
            direction="horizontal",
            gap=1,
        ),
        dbc.Stack(
            [
                dbc.Col("h", style=BOX_STYLE),
                dbc.Col("i", style=BOX_STYLE),
            ],
            direction="horizontal",
            gap=1,
        ),
        dbc.Stack(
            [
                dbc.Col("j", style=BOX_STYLE),
                dbc.Col("k", style=BOX_STYLE),
            ],
            direction="horizontal",
            gap=1,
        ),
        dbc.Stack(
            [
                dbc.Col("l", style=BOX_STYLE),
                dbc.Col("m", style=BOX_STYLE),
                dbc.Col("n", style=BOX_STYLE),
                dbc.Col("o", style=BOX_STYLE),
            ],
            direction="horizontal",
            gap=1,
        ),
    ],
    direction="vertical",
    gap=1,
    className="justify-content-center align-items-center",
)

app.layout = dbc.Container(
    [
        html.H1("Dash Grid Layout Example"),
        html.Hr(),
        dbc.Stack(
            [
                dbc.Stack(
                    [box1, box2, box3],
                    direction="horizontal",
                    gap=3,
                ),
                dbc.Stack(
                    [box4, box5, box6],
                    direction="horizontal",
                    gap=3,
                ),
            ],
            direction="vertical",
            gap=3,
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)

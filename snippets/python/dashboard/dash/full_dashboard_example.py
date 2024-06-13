"""
TODO

REQUIREMENTS: TODO (dash dash_auth dash_bootstrap_components pandas scipy) 
USAGE: $ python dash_multi_tab_app.py

Plan: 
    - the dashboard will generate 3 random datasets on launch, and will use these throughout the session
    - tabs on the dashboard:
        - a persistent continuously updated log of user's activity on the dashboard 
        - a raw view of the dataset currently selected (table)
        - a beautiful grid of visualisations of the selected dataset
    - user can select which of the 3 datasets they want to look at (this control at the top of every tab)
    - implement basic auth (username and password for access)
    - dashboard must look HEAT (visually) 
"""

import datetime
import random

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import dash_table, Input, Output, dcc, html
from dash_auth import BasicAuth

app = dash.Dash(
    external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True
)

BasicAuth(
    app, {"admin": "password"}, secret_key="It4cQgcRTMxfNp4hdgliBIZ6BTErcddYzo/b7UDN"
)


def datetime_now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


global_log_strings = [f"{datetime_now()} Started session"]
global_current_dataset_id = 1
global_current_page_url = "/"


def simulate_data(n_rows: int) -> dict[int, list[dict]]:
    """TODO"""
    datasets = {}
    for dataset_id in range(1, 4):
        datasets[dataset_id] = []
        for t in range(n_rows):
            for group, group_mean in (("A", 50), ("B", 40), ("C", 60)):
                datasets[dataset_id].append(
                    {
                        "time": t,
                        "group": group,
                        "amount": int(random.gauss(group_mean, 20)),
                    }
                )

    return datasets


data = simulate_data(n_rows=100)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "padding": "2rem 1rem",
    # "background-color": "#f8f9fa",
}

# the styles for the main content:
# to the right of the sidebar and add some padding.
CONTENT_STYLE = {
    "margin-left": "12rem",
    "margin-right": "1rem",
    "width": "50rem",
    "padding": "2rem 1rem",
}

PLOT_STYLE = {
    "paper_bgcolor": "black",  # Background color of the entire figure
    "plot_bgcolor": "black",  # Background color of the plotting area,
    "font": dict(color="white"),  # Text color
    "xaxis": dict(
        showgrid=False,
        linecolor="white",
        tickcolor="white",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
    ),  # X-axis style
    "yaxis": dict(
        showgrid=False,
        linecolor="white",
        tickcolor="white",
        title_font=dict(color="white"),
        tickfont=dict(color="white"),
    ),  # Y-axis style
    "title_font": dict(color="white"),  # Title text color
}

sidebar = dbc.Nav(
    [
        html.H2("Plotly Dash Example", className="display-4"),
        # html.Hr(),
        # html.P("some text here", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Welcome", href="/", active="exact"),
                dbc.NavLink("Raw Data", href="/data", active="exact"),
                dbc.NavLink("Data Visualisations", href="/dataviz", active="exact"),
                dbc.NavLink("Dashboard Activity Log", href="/log", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# content = html.Div(
content = dbc.Container(
    [
        dbc.Stack(
            [
                dbc.DropdownMenu(
                    label="Select Dataset",
                    menu_variant="dark",
                    children=[
                        dbc.DropdownMenuItem(
                            "Dataset 1",
                            id="select-dataset1",
                            n_clicks=0,
                        ),
                        dbc.DropdownMenuItem(
                            "Dataset 2",
                            id="select-dataset2",
                            n_clicks=0,
                        ),
                        dbc.DropdownMenuItem(
                            "Dataset 3",
                            id="select-dataset3",
                            n_clicks=0,
                        ),
                    ],
                ),
                dbc.Alert(
                    children=f"Dataset {global_current_dataset_id}",
                    id="selected-dataset-alert",
                    color="light",
                ),
            ],
            direction="horizontal",
            gap=3,
        ),
        dbc.Container(id="page-content"),
    ],
    style=CONTENT_STYLE,
)

app.layout = dbc.Container([dcc.Location(id="url"), sidebar, content], fluid=True)
# app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(
    Output("page-content", "children"),
    [
        Input("url", "pathname"),
        Input("select-dataset1", "n_clicks"),
        Input("select-dataset2", "n_clicks"),
        Input("select-dataset3", "n_clicks"),
    ],
)
def render_page_content(pathname, select_dataset1, select_dataset2, select_dataset3):
    global global_log_strings
    global global_current_dataset_id
    global global_current_page_url

    ctx = dash.callback_context
    if ctx.triggered_id in ("select-dataset1", "select-dataset2", "select-dataset3"):
        dataset_id = int(ctx.triggered_id[-1])
        if dataset_id != global_current_dataset_id:
            global_current_dataset_id = dataset_id
            global_log_strings = [
                f"{datetime_now()} Selected dataset {dataset_id}",
                html.Br(),
            ] + global_log_strings

    if pathname == "/":
        if global_current_page_url != pathname:
            global_current_page_url = pathname
            global_log_strings = [
                f"{datetime_now()} Visited Welcome page",
                html.Br(),
            ] + global_log_strings
        return html.P("Welcome text goes here")
    elif pathname == "/data":
        if global_current_page_url != pathname:
            global_current_page_url = pathname
            global_log_strings = [
                f"{datetime_now()} Visited Raw Data page",
                html.Br(),
            ] + global_log_strings
        # return html.Div(
        return dbc.Stack(
            [
                dbc.Col(
                    dbc.Button("Download CSV", id="download_csv_button", n_clicks=0),
                    width="auto",
                ),
                dcc.Download(id="download-csv"),
                dbc.Col(
                    dash_table.DataTable(
                        data[global_current_dataset_id],
                        style_as_list_view=True,
                        style_header={
                            "backgroundColor": "rgb(30, 30, 30)",
                            "color": "white",
                        },
                        style_data={
                            "backgroundColor": "rgb(0, 0, 0)",
                            "color": "white",
                        },
                    ),
                    style={"padding": "0 5vw 0 5vw"},
                ),
            ],
            direction="vertical",
            gap=3,
        )
    elif pathname == "/dataviz":
        if global_current_page_url != pathname:
            global_current_page_url = pathname
            global_log_strings = [
                f"{datetime_now()} Visited Data Visualisations page",
                html.Br(),
            ] + global_log_strings
        # return html.P("Data Visualisations")
        selected_dataset_df = pd.DataFrame(data[global_current_dataset_id])
        return dbc.Stack(
            [
                dbc.Col(
                    dcc.Graph(
                        figure=px.line(
                            selected_dataset_df,
                            x="time",
                            y="amount",
                            color="group",
                            title="Line Plots",
                        ).update_layout(**PLOT_STYLE)
                    )
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=px.bar(
                            selected_dataset_df,
                            x="time",
                            y="amount",
                            color="group",
                            title="Stacked Bar Chart",
                        ).update_layout(**PLOT_STYLE)
                    )
                ),
                dbc.Stack(
                    [
                        dbc.Col(
                            dcc.Graph(
                                figure=px.histogram(
                                    data[global_current_dataset_id],
                                    x="amount",
                                    # y="",
                                    color="group",
                                    marginal="box",
                                    title="Overlaid Histograms",
                                ).update_layout(**PLOT_STYLE)
                                # figure=ff.create_distplot(
                                #     [
                                #         [
                                #             x["amount"]
                                #             for x in data[global_current_dataset_id]
                                #             if x["group"] == group
                                #         ]
                                #         for group in ("A", "B", "C")
                                #     ],
                                #     ["A", "B", "C"],
                                # ).update_layout(**PLOT_STYLE)
                            ),
                            width=8,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                figure=px.pie(
                                    selected_dataset_df.groupby("group")
                                    .agg(sum_amount=("amount", "sum"))
                                    .reset_index(),
                                    values="sum_amount",
                                    names="group",
                                    title="Pie Chart",
                                ).update_layout(**PLOT_STYLE)
                            ),
                            width=4,
                        ),
                    ],
                    direction="horizontal",
                ),
            ],
            direction="vertical",
            gap=0,
        )
    elif pathname == "/log":
        if global_current_page_url != pathname:
            global_current_page_url = pathname
        return html.P(global_log_strings)
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(
    Output("selected-dataset-alert", "children"),
    [
        Input("url", "pathname"),
        Input("select-dataset1", "n_clicks"),
        Input("select-dataset2", "n_clicks"),
        Input("select-dataset3", "n_clicks"),
    ],
)
def show_selected_dataset(*args):
    ctx = dash.callback_context
    if ctx.triggered_id:
        return f"Dataset {ctx.triggered_id[-1]}"


@app.callback(
    Output("download-csv", "data"),
    Input("download_csv_button", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    global global_log_strings
    global_log_strings = [
        f"{datetime_now()} Downloaded dataset {global_current_dataset_id}",
        html.Br(),
    ] + global_log_strings
    csv_contents: str = (
        ",".join(data[global_current_dataset_id][0].keys())
        + "\n"
        + "\n".join(
            [
                ",".join(str(col) for col in row.values())
                for row in data[global_current_dataset_id]
            ]
        )
    )
    return dict(
        content=csv_contents, filename=f"dataset_{global_current_dataset_id}.csv"
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)

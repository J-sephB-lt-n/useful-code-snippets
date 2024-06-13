"""
TODO

REQUIREMENTS: TODO (dash dash_bootstrap_components pandas scipy) 
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

app = dash.Dash(
    external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True
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


def overlaid_histograms_fig(df) -> go.Figure:
    """Many thanks to ChatGPT for this code"""
    fig = go.Figure()

    # Add histograms for each group
    for group in df["group"].unique():
        group_data = df[df["group"] == group]
        fig.add_trace(
            go.Histogram(
                x=group_data["amount"],
                name=f"Histogram {group}",
                histnorm="probability density",
                opacity=0.5,
            )
        )

    # Update layout for better visibility
    fig.update_layout(
        barmode="overlay",
        title="Overlaid Histograms and Density Plots",
        xaxis_title="Amount",
        yaxis_title="Frequency",
        legend_title="Group",
        template="plotly_white",
    )

    return fig


data = simulate_data(n_rows=100)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    # "background-color": "#f8f9fa",
}

# the styles for the main content:
# to the right of the sidebar and add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# sidebar = html.Div(
sidebar = dbc.Nav(
    [
        html.H2("Plotly Dash Example", className="display-4"),
        html.Hr(),
        html.P("some text here", className="lead"),
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

content = html.Div(
    [
        html.Div(
            [
                dcc.Dropdown(
                    [
                        "Selected dataset: [dataset 1]",
                        "Selected dataset: [dataset 2]",
                        "Selected dataset: [dataset 3]",
                    ],
                    "Selected dataset: [dataset 1]",
                    id="dataset-selector",
                ),
            ],
            style=CONTENT_STYLE,
        ),
        html.Div(id="page-content", style=CONTENT_STYLE),
    ],
)

app.layout = dbc.Container([dcc.Location(id="url"), sidebar, content])
# app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname"), Input("dataset-selector", "value")],
)
def render_page_content(pathname, selected_dataset):
    global global_log_strings
    global global_current_dataset_id
    global global_current_page_url

    dataset_id = int(selected_dataset[-2])
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
        dataset_id = int(selected_dataset[-2])
        return html.Div(
            [
                html.Button("Download CSV", id="download_csv_button", n_clicks=0),
                dcc.Download(id="download-csv"),
                dash_table.DataTable(data[dataset_id]),
            ]
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
                dcc.Graph(
                    figure=px.line(
                        selected_dataset_df,
                        x="time",
                        y="amount",
                        color="group",
                        title="Line Plots",
                    )
                ),
                dcc.Graph(
                    figure=px.bar(
                        selected_dataset_df,
                        x="time",
                        y="amount",
                        color="group",
                        title="Stacked Bar Chart",
                    )
                ),
                dbc.Stack(
                    [
                        dbc.Col(
                            dcc.Graph(
                                figure=ff.create_distplot(
                                    [
                                        [
                                            x["amount"]
                                            for x in data[dataset_id]
                                            if x["group"] == group
                                        ]
                                        for group in ("A", "B", "C")
                                    ],
                                    ["A", "B", "C"],
                                )
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
                                )
                            ),
                            width=4,
                        ),
                    ],
                    direction="horizontal",
                    gap=3,
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

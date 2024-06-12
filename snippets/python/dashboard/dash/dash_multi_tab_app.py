"""
TAGS: dash|dashboard|data visualisation|data visualization|datavis|dataviz|frontend|gui|html|many|multi|multipage|multitab|page|pages|plotly|tab|tabs|ui|user interface|visualisation|visualization|web|website
DESCRIPTION: Illustration of a Dash app with multiple tabs (navbar on the left-hand side) 
REQUIREMENTS: pip install "dash==2.17.0" "dash-bootstrap-components==1.6.0"
USAGE: $ python dash_multi_tab_app.py
NOTES: I got this code from https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar
"""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content:
# to the right of the sidebar and add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P("A simple sidebar layout with navigation links", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page.")
    elif pathname == "/page-1":
        return html.P("This is the content of page 1.")
    elif pathname == "/page-2":
        return html.P("This is the content of page 2.")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)

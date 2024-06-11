"""
TAGS: auth|authenticate|authentication|basic|basic auth|basic authentication|dash|dashboard|data visualisation|data visualization|datavis|dataviz|frontend|gui|html|plotly|ui|user interface|visualisation|visualization|web|website
DESCRIPTION: Illustration of basic authentication (username + password) in a python Dash dashboard 
REQUIREMENTS: pip install "dash==2.17.0" "dash_auth==2.3.0"
USAGE: $ python dash_basic_auth.py
"""

from dash import Dash, html
from dash_auth import BasicAuth


# an alternative to an authorization function is to pass a list of
# username+passwords to BasicAuth()
def authorization_function(username, password):
    if (username == "admin") and (password == "StrongPassword"):
        return True
    else:
        return False


app = Dash(__name__)
BasicAuth(app, auth_func=authorization_function, secret_key="Vsqh%MDG3.&s#5e")

app.layout = [
    html.H1(
        children="You are successfully authenticated", style={"textAlign": "center"}
    ),
]

if __name__ == "__main__":
    app.run_server(debug=True)

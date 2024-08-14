```
TAGS: access|auth|authentication|authorisation|authorization|decorator|flask|protect|protected|token|wrapper
DESCRIPTION: Example of a decorator adding simple static token-based authorization around a Flask route
REQUIREMENTS: pip install Flask
NOTES: In a real application, access tokens should be specific to a user and managed securely
```

```python
# app.py
from typing import Optional
import flask
import auth

app = flask.Flask(__name__)

app.secret_key = b"\x97\xd8P\x1f\\\xa7\x9c\x13\x1ar,\x8e\x08\xbd\x94\x05\xfa\xc2[\x8d\xacCD\xa2"  # os.urandom(24)

app.config["USER_AUTH_DB"] = {
    # this is a toy example - please don't store passwords unhashed and use a proper database #
    "user1@email.com": {"password": "admin"},
    "user2@email.com": {"password": "password123!"},
}

@app.route("/am_i_in", methods=["GET"])
@auth.requires_access_token(required_token="abc123")
def am_i_in():
    """Tells user whether their access token cookie is working correctly"""
    return flask.Response("You are in", status=200)

@app.route("/log_in", methods=["POST"])
def log_in():
    """Client provides a username and password, and is provided with an access token cookie"""
    request_auth = flask.request.authorization
    if not request_auth or not request_auth.username or not request_auth.password:
        flask.abort(401)

    user_data: Optional[dict[str, str]] = app.config["USER_AUTH_DB"].get(
        request_auth.username
    )
    if user_data is None or request_auth.password != user_data.get("password"):
        return flask.abort(403)

    response = flask.make_response("OK")
    response.set_cookie(
        "access-token",
        "abc123",
        max_age=60 * 60 * 24,  # Cookie expiration time (1 day)
        httponly=True,  # Prevent JavaScript access
        secure=True,  # Only send cookie over HTTPS
        samesite="Lax",  # Prevent CSRF attacks
    )
    return response
```

```python
# auth.py
from functools import wraps
from typing import Optional
import flask

def requires_access_token(required_token: str):
    """A decorator wrapping authorisation functionality around a Flask route

    Args:
        required_token (str): The access token required for access
    """

    def decorator(func):
        @wraps(func)
        def decorated_function(*args, **kwargs):
            request_token: Optional[str] = flask.request.cookies.get("access-token")
            if not request_token:
                print("no token")
                flask.abort(401)
            if request_token != required_token:
                print("incorrect token")
                flask.abort(403)
            print("got to this place")
            return func(*args, **kwargs)

        return decorated_function

    return decorator
```

```
TAGS: api|flask|gunicorn|host|http|https|import|local|locally|module|programmatic|programmatically|server
DESCRIPTION: Locally host a Flask app server from another python script (allowing user keyboard-interrupt to stop the server)
REQUIREMENTS: pip install Flask gunicorn
```

```bash
tree
.
├── flask_app.py
├── main.py
└── run_local_flask_app.py

1 directory, 3 files
```

```python
# main.py
from run_local_flask_app import run_local_flask_app

if __name__ == "__main__":
    run_local_flask_app(flask_app_name="flask_app", port=6969)
```

```python
# run_local_flask_app.py
import subprocess
import time

def run_local_flask_app(flask_app_name: str, port: int) -> None:
    """Locally host a Flask app (until user keyboard-interrupt)

    Args:
        flask_app_name (str): Name of .py script containing the Flask app
        port (int): Port on which to (attempt to) host the app
    """
    process = subprocess.Popen(
        [
            "gunicorn",
            "-b",
            f"0.0.0.0:{port}",
            "--worker-class=sync",
            f"{flask_app_name}:app",
        ]
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"Received manual interrupt, shutting down Flask server '{flask_app_name}'")
    finally:
        process.terminate()
        process.wait()
```

```python
# flask_app.py
import flask

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return flask.Response("OK", status=200)
```

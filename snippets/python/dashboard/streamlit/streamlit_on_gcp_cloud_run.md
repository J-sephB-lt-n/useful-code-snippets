```
TAGS: cloud|cloud run|dashboard|data visualisation|data visualization|datavis|dataviz|frontend|gcp|google|google cloud|gui|html|iframe|streamlit|ui|user interface|visualisation|visualization|web|website
DESCRIPTION: Deploy a streamlit dashboard on a Google Cloud Run service, and include it in an HTML page as an iframe
```

Directory structure:

```bash
streamlit_on_gcp_cloud_run/
├── Dockerfile
├── build_deploy_cloud_run.sh
├── iframe_test.html
├── requirements.txt
└── streamlit_app.py
```

Deploy from terminal:

```bash
GCP_PROJ_ID="your project ID"
GCP_REGION_NAME="desired region for your cloud run service e.g. europe-west12"
GCP_ARTIFACT_REG_REPO_NAME="name of existing docker repository in GCP artifact registry"
API_NAME="desired name for cloud run service"
source build_deploy_cloud_run.sh
```

```dockerfile
# Dockerfile
FROM python:3.12-slim
EXPOSE 8080
WORKDIR /streamlit_on_gcp_cloud_run
COPY . ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

```bash
# build_deploy_cloud_run.sh
echo "start time: $(date)"
gcloud config set project $GCP_PROJ_ID
gcloud config set run/region $GCP_REGION_NAME
docker_image_name=$GCP_REGION_NAME-docker.pkg.dev/$GCP_PROJ_ID/$GCP_ARTIFACT_REG_REPO_NAME/$API_NAME
echo "started build: $(date)"
build_start_time=$(date +%s)
gcloud builds submit --tag $docker_image_name
build_end_time=$(date +%s)
echo "started deploy: $(date)"
deploy_start_time=$(date +%s)
gcloud run deploy $API_NAME \
    --image $docker_image_name \
    --max-instances 1 \
    --min-instances 0 \
    --allow-unauthenticated \
    --timeout 600 \
    --ingress all
deploy_end_time=$(date +%s)
echo "finished: $(date)"
```

```html
# iframe_test.html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />
    <title>Streamlit iframe Test</title>
  </head>
  <body>
    <main>
      <h1>Streamlit iframe Test</h1>
      <iframe
        src="https://yourDeployedCloudRunUrlHere/"
        width="100%"
        height="500px"
      ></iframe>
    </main>
  </body>
</html>
```

```
# requirements.txt
streamlit==1.35.0
```

```python
# streamlit_app.py
# (this is just the example from the streamlit documentation)
import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber pickups in NYC")

DATE_COLUMN = "date/time"
DATA_URL = (
    "https://s3-us-west-2.amazonaws.com/"
    "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text("Loading data...")
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(data)

st.subheader("Number of pickups by hour")
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider("hour", 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader("Map of all pickups at %s:00" % hour_to_filter)
st.map(filtered_data)
```

```
TAGS: api|build|cloud run|deploy|endpoint|flask|gcloud|gcp|google|google cloud|gunicorn|iac|infrastructure|lambda|microservice|microservices|serverless|rest|run
DESCRIPTION: Boilerplate code for deploying a python Flask app on Google Cloud Run service
```

```bash
tree .
.
├── Dockerfile
├── Makefile
├── app.py
├── cleanup_policy_artifact_registry.json
├── requirements.txt
```

```bash
# terminal commands
GCP_PROJ_ID='your-gcp-proj-id'
GCP_REGION='europe-west2'
CREATE_ARTIFACT_REG_REPO_NAME='your-artifact-registry-repo-name'
CREATE_CLOUD_RUN_SERVICE_NAME='your-cloud-run-service-name'

gcloud auth login
gcloud config set project $GCP_PROJ_ID # verify with `gcloud config get project`
gcloud config set run/region $GCP_REGION # verify with `gcloud config get run/region`

DOCKER_IMG_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJ_ID}/${CREATE_ARTIFACT_REG_REPO_NAME}/${CREATE_CLOUD_RUN_SERVICE_NAME}"

make create_artifact_registry_repo \
  ARTIFACT_REG_REPO_NAME=$CREATE_ARTIFACT_REG_REPO_NAME \
  GCP_REGION=$GCP_REGION \
  CLOUD_RUN_SERVICE_NAME=$CREATE_CLOUD_RUN_SERVICE_NAME

make build \
  DOCKER_IMG_URI=$DOCKER_IMG_URI

make deploy \
  CLOUD_RUN_SERVICE_NAME=$CREATE_CLOUD_RUN_SERVICE_NAME \
  DOCKER_IMG_URI=$DOCKER_IMG_URI
```

```docker
# Dockerfile
FROM python:3.12-slim

WORKDIR /cloud_run_app

# copy files into the container image #
COPY app.py app.py
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn \
  --bind :$PORT \
  --workers 1 \
  --threads 8 \
  --timeout 0 \
  app:app
```

```makefile
# Makefile
create_artifact_registry_repo:
  @clear
  gcloud artifacts repositories create $(ARTIFACT_REG_REPO_NAME) \
    --repository-format docker \
    --location $(GCP_REGION) \
    --description='Container image used by Cloud Run Service $(CLOUD_RUN_SERVICE_NAME)'
  gcloud artifacts repositories set-cleanup-policies $(ARTIFACT_REG_REPO_NAME) \
    --policy cleanup_policy_artifact_registry.json \
    --location=$(GCP_REGION)

build:
  @clear
  @echo $$(date +"%Y-%m-%d %H:%M:%S")' Started Docker image build (and push to artifact registry)'
  gcloud builds submit --tag '$(DOCKER_IMG_URI)'
  @echo $$(date +"%Y-%m-%d %H:%M:%S")' Finished Docker image build (and push to artifact registry)'

deploy:
  @clear
  @echo $$(date +"%Y-%m-%d %H:%M:%S")' Started deploying Cloud Run service'
  gcloud run deploy $(CLOUD_RUN_SERVICE_NAME) \
    --image '$(DOCKER_IMG_URI)' \
    --max-instances 4 \
    --min-instances 0 \
    --allow-unauthenticated \
    --timeout 30
  @echo $$(date +"%Y-%m-%d %H:%M:%S")' Finished deploying Cloud Run service'
```

```python
# app.py
"""The entrypoint of the Flask app"""

import flask

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def base():
    """An example endpoint"""
    return flask.Response("OK", status=200)
```

```json
-- cleanup_policy_artifact_registry.json --
[
  {
    "name": "keep-latest-2",
    "action": { "type": "Keep" },
    "mostRecentVersions": {
      "keepCount": 2
    }
  }
]
```

```bash
# requirements.txt
Flask
gunicorn
```

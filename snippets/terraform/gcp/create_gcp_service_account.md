```
TAGS: account|build|gcloud|gcp|google|google cloud|iac|infrastructure|service|service account|terraform
DESCRIPTION: terraform code for creating a GCP service account and adding IAM roles to it
```

```terraform
// main.tf //

/*
How to use this code:

1. Change any variable definitions that you want to (e.g. service account name, GCP project ID etc.)
2. Run the following in terminal:
  - $ terraform init # once-off setup of this terraform project
  - $ terraform plan # terraform explains what it would do if you were to run `terraform apply`
  - $ terraform apply # make the infrastructure changes
*/

variable "gcp_project_id" {
  type        = string
  default     = "your-gcp-project-id"
  description = "ID of Google Cloud Platform project"
  nullable    = false
}

provider "google" {
  project = var.gcp_project_id
  region  = "europe-west2"
  zone    = "europe-west2-c"
}

// Create the service account
resource "google_service_account" "service_account" {
  account_id   = "name-appearing-in-uri"
  display_name = "Name shown in GCP GUI"
  description  = "Description of what this service account will be used for"
}

// Assign roles to the service account
resource "google_project_iam_member" "service_iam_roles" {
  depends_on = [google_service_account.service_account]

  for_each = toset([
    // just comment out the roles that you don't want
    "roles/bigquery.dataEditor",          # allow read and update table/view data/metadata. Allows delete table/view.
    "roles/bigquery.dataViewer",          # read only table/view data/metadata
    "roles/bigquery.jobUser",             # can run BigQuery jobs
    "roles/bigquery.user",                # read/write tables/table-metadata, list tables & datasets, create/list/run/delete jobs, create tables & datasets
    "roles/cloudscheduler.jobRunner",     # allows execute/list/view existing jobs in cloud scheduler
    "roles/run.invoker",                  # allows HTTP requests to a cloud run service (or cloud run jobs)
    "roles/logging.logWriter",            # allows writing log messages to GCP Cloud Logging
    "roles/secretmanager.secretAccessor", # read only cloud secrets
    "roles/storage.objectAdmin",          # full control over objects in storage bucket
    "roles/storage.objectCreator",        # can create objects in bucket but not overwrite/view/delete
    "roles/storage.objectUser",           # create/read/update/delete (CRUD) objects in bucket
    "roles/storage.objectViewer",         # can only view objects and their metadata in bucket
  ])

  project = var.gcp_project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.service_account.email}"
}
```

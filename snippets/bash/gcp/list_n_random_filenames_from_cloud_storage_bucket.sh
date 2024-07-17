<<'###BLOCK-COMMENT'
TAGS: bucket|cli|cloud|cloud storage|gcloud|gcloud cli|gcloud storage|gcp|google|google cloud|list|n|object|objects|random|sample|selection
DESCRIPTION: Returns the names of `n` random files from specified bucket, matching the filter pattern provided   
###BLOCK-COMMENT

# this is giving 20 random object names #
gcloud storage ls \
  --project your-gcp-project-id \
  --recursive \
  'gs://your-bucket-name/a/folder/**.json' | sort -R | head -n 20

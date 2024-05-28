#!/bin/bash
gcloud dataproc clusters delete \
    $SPARK_CLUST_NAME \
    --region europe-west2 \
    --quiet

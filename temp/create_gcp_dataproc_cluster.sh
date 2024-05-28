#!/bin/bash
gcloud dataproc clusters create \
    $SPARK_CLUST_NAME \
    --region europe-west2 \
    --zone europe-west2-c \
    --enable-component-gateway \
    --master-machine-type n2-standard-4 \
    --master-boot-disk-type pd-balanced \
    --master-boot-disk-size 500 \
    --num-workers $N_WORKERS \
    --worker-machine-type n2-standard-4 \
    --worker-boot-disk-type pd-balanced \
    --worker-boot-disk-size 500 \
    --image-version 2.2-debian12 \
    --project $GCP_PROJ_NAME

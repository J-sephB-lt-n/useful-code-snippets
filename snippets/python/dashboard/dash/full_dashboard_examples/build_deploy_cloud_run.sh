echo "start time: $(date)"
gcloud config set project $GCP_PROJ_ID
gcloud config set run/region $GCP_REGION_NAME
docker_image_name=$GCP_REGION_NAME-docker.pkg.dev/$GCP_PROJ_ID/$GCP_ARTIFACT_REG_REPO_NAME/$API_NAME
echo "started build: $(date)"
gcloud builds submit --tag $docker_image_name
echo "started deploy: $(date)"
gcloud run deploy $API_NAME \
	--image $docker_image_name \
	--max-instances 1 \
	--min-instances 0 \
	--allow-unauthenticated \
	--timeout 600 \
	--ingress all
echo "finished: $(date)"

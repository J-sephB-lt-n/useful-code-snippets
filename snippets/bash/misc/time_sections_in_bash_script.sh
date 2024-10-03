<<'###BLOCK-COMMENT'
TAGS: time|timer|section|sections
DESCRIPTION: Code to time different parts of a bash script 
###BLOCK-COMMENT

echo "started process: $(date)"
process_start_time=$(date +%s)
sleep 1

echo "started build: $(date)"
build_start_time=$(date +%s)
sleep 2
build_end_time=$(date +%s)
echo "finished build: $(date)"

echo "started deploy: $(date)"
deploy_start_time=$(date +%s)
sleep 3
deploy_end_time=$(date +%s)
echo "finished deploy: $(date)"

process_end_time=$(date +%s)
echo "finished process: $(date)"
echo "Build took "$((build_end_time - build_start_time))" seconds"
echo "Deployment took "$((deploy_end_time - deploy_start_time))" seconds"
echo "Total process took "$((process_end_time - process_start_time))" seconds"

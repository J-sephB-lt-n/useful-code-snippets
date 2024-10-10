```
TAGS: build|container|docker|image|local|run
DESCRIPTION: commands to build and run a docker container locally (e.g. for dev)
```

```shell
# run these commands in the same directory as the Dockerfile
docker build --tag temp .
docker run -it --name temprun temp bash
docker stop temprun
docker rm temprun
```

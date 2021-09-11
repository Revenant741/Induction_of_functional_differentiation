#!/bin/bash

d_tag=$1
d_tag_split=(${d_tag//:/ })
d_tag_name=${d_tag_split[0]}
d_image=${USER}/${d_tag}

echo "s"
echo $d_tag
echo $d_tag_name
echo $d_image
echo "e"

docker container rm $d_tag_name
docker container ls -a
docker image rm $d_image
docker image prune -f
docker image ls

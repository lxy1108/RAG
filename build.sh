#!/bin/bash

# 以下命令作为打包示例，实际使用时请修改为自己的镜像地址

URL=registry.cn-shanghai.aliyuncs.com

IMAGE_NAME=monkey

VERSION=0.1

docker build -t $URL/$IMAGE_NAME:$VERSION .





# FROM python:3.9.7-buster
FROM docker.io/library/python:3.7
ARG DEBIAN_FRONTEND=noninteractive

# パッケージの追加とタイムゾーンの設定
# 必要に応じてインストールするパッケージを追加してください
RUN apt-get update && apt-get install -y \
    tzdata \
&&  apt-get -y install gcc libmariadb-dev \
&&  ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
&&  apt-get clean \
&&  rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

COPY ./requirements.txt /tmp
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir /app
RUN mkdir /app/py3
RUN mkdir /app/py3/module
RUN mkdir /app/downlaod

COPY ./module /app/py3/module 

WORKDIR /app/py3/module
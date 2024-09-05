FROM ubuntu:22.04
LABEL author="Abhijith Ganesh - On behalf of Team APP"

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip sqlite3 libsqlite3-dev python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install -r requirements.txt
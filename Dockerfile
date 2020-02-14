FROM python:3.6.9

ENV PYTHONUNBUFFERED 1

RUN mkdir /vector_api
WORKDIR /vector_api

ADD requirements.txt /vector_api/

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ADD . /vector_api/


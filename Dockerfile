FROM tensorflow/tensorflow
# FROM jupyter/scipy-notebook:17aba6048f44
WORKDIR /all
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
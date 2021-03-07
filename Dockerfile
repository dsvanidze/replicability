FROM continuumio/miniconda3:4.9.2
WORKDIR /all
COPY setup.py ./
COPY src/ ./src/
COPY environment.yml ./
SHELL ["/bin/bash", "-c"]
ENV BASH_ENV ~/.bashrc
# RUN conda update conda
RUN conda install mamba -n base -c conda-forge
RUN mamba env create -f environment.yml
RUN conda init bash
RUN source "../root/.bashrc"
RUN echo "conda activate replicability" >> ~/.bashrc
RUN echo "jpt() { if [[ \$1 == start ]]; then jupyter lab --ip=0.0.0.0 --port=\$2 --no-browser --allow-root & elif [ \$1 = stop ]; then jupyter notebook stop \$2; fi }" >> ~/.bashrc
#RUN conda config --env --set channel_priority strict
#RUN conda config --env --add channels conda-forge
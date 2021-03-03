FROM conda/miniconda3
WORKDIR /all
COPY setup.py ./
COPY src/ ./src/
COPY environment.yml ./
SHELL ["/bin/bash", "-c"]
ENV BASH_ENV ~/.bashrc
RUN conda update conda
RUN conda env create -f environment.yml
RUN conda init
RUN source "../root/.bashrc"
RUN echo "conda activate bachelor" >> ~/.bashrc
RUN echo "jpt() { if [[ \$1 == start ]]; then jupyter lab --ip=0.0.0.0 --port=\$2 --no-browser --allow-root & elif [ \$1 = stop ]; then jupyter notebook stop \$2; fi }" >> ~/.bashrc
#RUN conda config --env --set channel_priority strict
#RUN conda config --env --add channels conda-forge
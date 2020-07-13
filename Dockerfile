FROM conda/miniconda3
WORKDIR /all
COPY environment.yml ./
RUN conda update conda
RUN conda env create -f environment.yml
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN conda init
RUN echo "conda activate bachelor" >> ~/.bashrc

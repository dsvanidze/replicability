# Load miniconda image
FROM continuumio/miniconda3:4.9.2
# Set working directory to /all
WORKDIR /all
# Copy set.py into Docker environment
COPY setup.py ./
# Copy src directory into Docker environment
COPY src/ ./src/
# Copy conda enrionment file into Docker environment
COPY environment.yml ./
# Set default command line to "bash"
SHELL ["/bin/bash", "-c"]
ENV BASH_ENV ~/.bashrc
# Install mamba package manager (almost same as conda) 
# for resolving package dependencies faster than conda (https://github.com/mamba-org/mamba)
RUN conda install mamba -n base -c conda-forge
# Create a mamba environment and install all listed packages from environment.yml file
RUN mamba env create -f environment.yml
# Initialise conda environment (created by mamba) in bash
RUN conda init bash
RUN source "../root/.bashrc"
# Save activation of the conda environment for the project in the bash config file
# In order to activate the enironment everytime bash starts within Docker
RUN echo "conda activate replicability" >> ~/.bashrc
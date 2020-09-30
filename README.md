# Fine-Scale Spatial Predictions of COVID-19 Cases in China using GIS Data and Deep Learning Algorithms

This repository provides source code and replication guide of the study in PDF.


## Reproduction steps

### Step 1: Download project
To easily get the project files locally, you can install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and use the clone command `git clone https://github.com/dsvanidze/fine-scale-spatial-predictions-of-covid-19-cases-in-china-using-gis-data-and-deep-learning.git` in your command-line tool or just follow the [Github guide for downloading files](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository).

<br>

### Step 2: Install Docker and Docker Compose
Official Docker documentation explains how to install [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) for different operating systems (Mac, Windows, Linux).

<br>

### Step 3: Execute the code locally
After following the steps before, you will have a project folder locally with the name *fine-scale-spatial-predictions-of-covid-19-cases-in-china-using-gis-data-and-deep-learning*. 

<br>
 
You can navigate to the folder with your command-line tool with the command `cd <PATH_TO_THE_FOLDER>/fine-scale-spatial-predictions-of-covid-19-cases-in-china-using-gis-data-and-deep-learning` where <PATH_TO_THE_FOLDER> would be the absolute path to the folder.

<br>

After you  navigate to  the folder you can build and run the Docker container with the command `docker-compose up -d --build` (without --build if you ran it already). You can then use the command-line in your Docker container to have access to all installed tools and project requirements internally. For this, use the command `docker-compose exec bachelor bash`.

<br>

Finally, if you want to run Jupyter notebooks, you can run `jpt start 9008` in the command-line tool of the Docker container . This will start a Jupyter notebook on the port 9008. After starting Jupyter Notebook, it will log a link in the command-line tool like [http://127.0.0.1:9009/?token=917d4ee70903755d534ef613ff9eac9113d079985a383113](http://127.0.0.1:9009/?token=917d4ee70903755d534ef613ff9eac9113d079985a383113). You can copy the link in your browser and access the Jupyter notebook running in the Docker container.

<br>

This is all. Now you have the exact same code, with the exact same tools and requirements as I used for the project development. You did not need to install Python, Conda or other dependencies for the project.
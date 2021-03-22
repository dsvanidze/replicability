## Towards Replicability and Reproductibility in Geography: A General Framework for Predictive Modeling on Spatial Data

This repository provides source code, replication framework and guide of a forthcoming paper.


## Reproduction steps

### Step 1: Requirements
To easily get the project files locally, you can download the project .ZIP locally from [here](https://github.com/dsvanidze/replicability/archive/refs/heads/master.zip) and extract it. Alternatively, if you already use git, follow the [Github guide for downloading files](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository).

<br>

In addition, official Docker documentation explains how to install [Docker](https://docs.docker.com/get-docker/) for different operating systems (Mac, Windows, Linux).

### Step 2: Build Docker container
After following the steps before, you will have a project folder locally with the name *replicability-master* (or *replicability* if used git). Get an absolute path to the folder and navigate to it within command line:

`cd absolute_path_to_folder`

<br>

After you  navigate to  the folder you can build the project Docker container, which will take several minutes to automatically install all project requirements:

`docker-compose build`

### Step 3: Work with project
To run Docker container and work with the project, you will only need to run:

`docker-compose up`

<br>

Now you can interact with the project by copying Jupyter notebook link into your browser, which looks like this in the command-line output:

[http://127.0.0.1:9009/?token=917d4ee70903755d534ef613ff9eac9113d079985a383113](http://127.0.0.1:9009/?token=917d4ee70903755d534ef613ff9eac9113d079985a383113)

### Conclusion

This is all. Now you have the exact same code, with the exact same tools and requirements as we, collaborators, use for the project development. You do not need to install Python, Conda or other dependencies for the project, it is already done for you automatically.

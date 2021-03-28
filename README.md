## Towards Replicability and Reproductibility in Geography: A General Framework for Predictive Modeling on Spatial Data

This repository provides source code, replication framework and guide of a forthcoming paper.


## Reproduction steps

### Step 1: Requirements

#### 1.1. Download project locally
To easily get the project files locally, you can download the project .ZIP locally from [here](https://github.com/dsvanidze/replicability/archive/refs/heads/master.zip) and extract it. Alternatively, if you already use git, follow the [Github guide for downloading files](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository).

#### 1.2. Install Docker
Docker installation file is about 500MB. On average with 0.7MB/s internet speed its download would take 12 minutes. Its documentation explains step by step, how to download and install [Docker Desktop MacOS](https://docs.docker.com/docker-for-mac/install/), [Docker Desktop Windows](https://docs.docker.com/docker-for-windows/install/) and [Docker Server Linux](https://docs.docker.com/engine/install/#server). After Docker is installed, the machine needs to be restarted. After restart you need to open Docker software. To check if Docker runs and is sucessefully installed, running this in command line `docker --version` should output the version of Docker. 

In case:
* you are located in China, please download first Docker installation file from [Daocloud.io](http://get.daocloud.io/#install-docker-for-mac-windows) and then follow all other steps for [MacOS](https://docs.docker.com/docker-for-mac/install/), [Windows](https://docs.docker.com/docker-for-windows/install/) and [Linux](https://docs.docker.com/engine/install/). 
* your OS is Windows and Docker asks you for WSL2 Linux Kernel, you can download and run the installation file [Windows Linux Kernel](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi). If you have an ARM64 machine, you are required to download and run the installation file [ARM64 Package](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_arm64.msi). If you are not sure, which type of machine you have, type in command line `systeminfo | find "System Type"`. Installation of WSL2 is very fast and can take only a few seconds. After this you need to follow Docker instructions and restart your machine.
* your machine is Windows and you installed Windows Linux Kernel from the previous step and starting of Docker software take very long (3+ Minutes). Fix it by running `wsl --unregister docker-desktop` and `wsl --unregister docker-desktop-data` in command line (source: [Stackoverflow](https://stackoverflow.com/a/62495664/6072503)).


In addition, official Docker documentation explains how to install [Docker](https://docs.docker.com/get-docker/) for different operating systems (Mac, Windows, Linux).

### Step 2: Build Docker container
After following the steps before, you will have a project folder locally with the name *replicability-master* (or *replicability* if used git). To get the absolute path and navigate to the folder:
1. Open command line by searching for Terminal on MacOS/Linux and cmd on Windows machines
2. Type in command line `cd '`
4. Drag the folder to the command line
5. Type after a path is added `'`

As a result, for example, you will see in command line of MacOS/Linux:

`cd '/Users/dsvanidze/Desktop/replicability-master'`

Or, in command line of Windows:

`cd '"C:\Users\dsvanidze\Desktop\replicability-master"'`

5. Press Enter and it will navigate automatically in command line to the folder

After you  navigate to  the folder you can build the project Docker container, which will take about 5 minutes to automatically install all project requirements:

`docker-compose build`

### Step 3: Work with project
To run Docker container and work with the project, you will only need to run:

`docker-compose up`

This only takes about 5 seconds. Now you can interact with the project by copying Jupyter notebook link into your browser. Jupyter notebok will provide three different links, but only the last one will work for you, which looks like this in the command-line output:

[http://127.0.0.1:9009/?token=917d4ee70903755d534ef613ff9eac9113d079985a383113](http://127.0.0.1:9009/?token=917d4ee70903755d534ef613ff9eac9113d079985a383113)

After Jupyter runs in your browser, you can navigate across folders and files in Jupyter by a double click on them. You can open and run Jupyter notebook in notebooks folder. For example, this [Jupyter notebook](http://127.0.0.1:9009/lab/tree/notebooks/predictions/generate-fine-scale-features.ipynb) provides fine-scale predictions of COVID-19 cases.

### Conclusion

This is all. Now you have the exact same code, with the exact same tools and requirements as we, collaborators, use for the project development. You do not need to install Python, Conda or other dependencies for the project, it is already done for you automatically.

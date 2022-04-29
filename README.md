## Towards Replicable Deep Learning Algorithms in Public Health: A Case Study on Predicting Early COVID-19 Infections in China

This repository provides source code, replication framework and guide of our research work. You can view the [project online](http://c100-159.cloud.gwdg.de:9009/lab/tree/notebooks/main.ipynb?token=7cf55c2887d81e8ea8da627112d0753e4b4fc79345f121fc) (read-only) or reproduce it locally (editable).


## Reproduction steps

### Step 1: Requirements

#### 1.1. Download project locally
To work on your local machine, you can download the project .ZIP file locally from [here](https://github.com/dsvanidze/replicability/archive/refs/heads/master.zip). Alternatively, if you already use git, follow the [Github guide for downloading files](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository).

#### 1.2. Install Docker
Docker installation file is about 500MB. On average with a 0.7MB/s internet speed, it should take about 12 minutes to download the file. Its documentation explains step by step, how to download and install [Docker Desktop MacOS](https://docs.docker.com/docker-for-mac/install/), [Docker Desktop Windows](https://docs.docker.com/docker-for-windows/install/) and [Docker Server Linux](https://docs.docker.com/engine/install/#server). After the installation of Docker, it is important to restart your local machine. After restarting your machine, you will need to open the Docker software. To check if Docker runs and is successfully installed, running this in the command line `docker --version` should output the version of Docker. You can open the command line by searching for Terminal on MacOS/Linux and cmd on Windows machines.

Additional remarks:
* if you are located in China, please download first Docker installation file from [Daocloud.io](http://get.daocloud.io/#install-docker-for-mac-windows) and then follow the steps for [MacOS](https://docs.docker.com/docker-for-mac/install/), [Windows](https://docs.docker.com/docker-for-windows/install/) and [Linux](https://docs.docker.com/engine/install/).
* if your operating system (OS) is Windows and Docker asks you to get WSL2 Linux Kernel. If so, you will need to download and run the installation file [Windows Linux Kernel](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi). If you have an ARM64 machine, you are required to download and run the installation file [ARM64 Package](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_arm64.msi). If you are not sure about the type of machine you have, type in the command line `systeminfo | find "System Type"`. The installation of WSL2 is very fast and should only take a few seconds. After this step, please follow the instructions from the software Docker and restart your machine.
* if your OS is Windows and you have installed WSL2 Linux Kernel from the previous step and you observe that Docker software has not completed its starting operation (if it takes more than 3 minutes and Docker is still not running, this means that the software has an issue with the registration process. To fix the issue, run `wsl --unregister docker-desktop` and `wsl --unregister docker-desktop-data` in command line (source: [Stackoverflow](https://stackoverflow.com/a/62495664/6072503)).

### Step 2: Build Docker container
After following the steps before, you will have a project folder generated on your local machine with the name *replicability-master* (or *replicability* if used git). To get the absolute path and navigate to the folder:
1. Type in command line `cd '`
1. Drag the folder to the command line
1. Type after a path is added `'`

As a result, for example, you will see in the command line of MacOS/Linux:

`cd '/Users/dsvanidze/Desktop/replicability-master'`

Or, in command line of Windows:

`cd '"C:\Users\dsvanidze\Desktop\replicability-master"'`

4. Press Enter and it will navigate automatically in the command line to the folder

After you  navigate to  the folder you can build the project Docker container, which will take about 5 minutes (depending on your machine) to automatically install all project requirements. To do so, type in the command line:

`docker-compose build`

### Step 3: Work with project
To run Docker container and work with the project, type in the command line:

`docker-compose up`

This only takes about 5 seconds. Now you can interact with the project by copying the Jupyter notebook link into your browser (you can use any browser, e.g. Google Chrome or Firefox). Jupyter notebok will provide three different links, but only the last one is likely to work for you. You should follow a link in the command-line output that looks like this:

[http://127.0.0.1:9009/?token=917d4ee70903755d534ef613ff9eac9113d079985a383113](http://127.0.0.1:9009/?token=917d4ee70903755d534ef613ff9eac9113d079985a383113)

Once Jupyter runs in your browser, you can navigate across the folders and files in Jupyter by double-clicking on them. You can open and run Jupyter notebook in the notebooks folder. For example, this [Jupyter notebook](http://127.0.0.1:9009/lab/tree/notebooks/predictions/generate-fine-scale-features.ipynb) provides fine-scale predictions of COVID-19 cases.

### Conclusion

This is all you need to replicate the work from this research work. Now you have the exact same code, with the exact same tools and requirements as we, collaborators, use for the project development. You do not need to install Python, Conda or other dependencies for the project, it is already done for you automatically.

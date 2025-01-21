# accessopp: Access to opportunities calculator

**accessopp** is a library that allows calculation of access to opportunities, a 
commonly used measure in transportation planning and urban design. This library 
is designed to support analyses being conducted by City of Toronto staff, 
although others are welcome to use.

The hardest part of calculating access to opportunities is to find calculate
the travel times, by mode. There are many ways to do this, and the best 
approach will depend on the application.

For calculating accessibilities from travel demand forecasts, such as from the
GTAv4 model, time matrices can be output from those runs and read directly
into this calculator. 

For present-day or near-future work, it will likely be adviseable to use
one of the available open-source routing packages to calculate travel times.
These libraries have the advantage of using detailed networks from the 
OpenStreetMap consortium, and calculate exact (latitude, longitude) origins and
destinations, allowing for more precise access calculations. The currently 
supported open-source packages are *OpenTripPlanner* (OTP), *r5py* and Valhalla.


## Installation

This installation guide is designed primarily for City of Toronto employees to 
install this library and its dependencies on (Windows) City infrastructure. 
This may vary slightly for other users.


### 1. Download miniconda package and environment management system. 

People working for the City of Toronto must use Miniconda instead of the more 
complete-featured Anaconda.

The latest version of miniconda can be found here.
https://docs.conda.io/projects/miniconda/en/latest/index.html

Install for a single-user (just me option), as to install for all users requires 
admin priviledges. At the time of writing, the default installation folder 
is `C:\Users\[user name]\AppData\Local\miniconda3`. You can use the default 
advanced installation options

### 2. Download *accesto*

#### Download `git` version control system -- optional
It is recommended, but likely not necessary to download the 'git` executable.
This will simplify receiving updates to this package.

Download and install a suitable git executable from 
[git downloads](https://git-scm.com/downloads). git currently does not 
require admin privileges. 

If using git on Windows on City of Toronto machines, we've found 
that we had to enter the following command in git before cloning the repository 
Thanks [StackOverflow](https://stackoverflow.com/questions/23885449/unable-to-resolve-unable-to-get-local-issuer-certificate-using-git-on-windows) 
for the tip.

```console
git config --global http.sslbackend schannel
```

Then download the code from this release and store in a convenient location.  
First go to the folder that you would like to hold this directory.

If using git, right-click the folder, select 'Git BASH Here'. Run the following
in Git Bash:
```console
$ git clone https://github.com/bryce-sharman/accessopp.git
```

If not using git, then you'll need to download and unzip the git file,
which you can download from this GitHub repository by clicking the green 
`<> Code` button at the top of the root page of this repository on GitHub, 
and selecting the `Download ZIP` option.

### 3. Install *accessopp*

Installing the package *accessopp* is a two-step process. The first step
is to create a conda environment that contains all dependies, the second
is to install *accessopp* itself.

**3.1: Create a conda environment with required dependencies.**

First run miniconda. From the Windows Start Menu, open 
*Anaconda Prompt (Miniconda3)*. Then change to the root directory of the 
downloaded *accessopp* package.

*accessopp* contains a conda *environment.yml* file that defines all
required dependencies. The environment name, *accessopp* is defined in the 
environment.yml file. To create a conda environment with the required 
dependencies, simply run the following command.

```console
> conda env create -f environment.yml
```

**3.2 Install *accessopp*
First, activate the new conda environment. You will need to do this step
whenever using the package.

```console
(base) conda activate accessopp
(accessopp) cd [path to root folder of this library] 
```

Then the following command to install this package.
```console
(accessopp) [local directory]> pip install --no-build-isolation --no-deps install -e .
```

> **_NOTE:_** Note that the root folder of the library contains the 
*pyproject.toml* file. The `--no-build-isolation` `--no-deps` are required so 
that the pip install doesn't try to modify the dependencies. 


To update this package, replace the previous command with the following:
```console
(accessopp) [local directory]> pip install --no-build-isolation --no-deps --upgrade -e .
```

### 4. Install Java

Java is required by *OpenTripPlanner*, *r5py* and *osmium*, which can be 
used to read, write and modify OpenStreetMap files.  If the system Java is too 
old then you will to install a new JDK.

OpenTripPlanner version 2.5, requires JDK 17. The following JDK was used during 
development and testing of this library. At the time of developing this 
package
https://learn.microsoft.com/en-us/java/openjdk/download#openjdk-17


> **_NOTE:_** JDK installation requires admin priviledges.


### 5. Download OpenTripPlanner JAR file (only if using OpenTripPlanner)

*OpenTripPlanner* is distributed as a single stand-alone runnable JAR file. 
OTP v2.4.0 can be downloaded from their GitHub repository.
https://github.com/opentripplanner/OpenTripPlanner/releases

Download the 'shaded jar' file, *e.g.* **otp-2.4.0-shaded.jar**.

**_NOTE:_** OTP v2.5.0 is now available, however this version requires Java 21. 
Testing for `accessopp` has only been completed using OTP v2.4.0. 


### 6. Download *r5* JAR file and setup the `r5py`` config file (only if using r5py)

On City of Toronto computers, we've found that it's best to run using a local 
copy of the r5 .jar file. 

Download the `r5` JAR file, which is the package that performs the travel time 
computations. This is available on *r5's* GitHub page: 
https://github.com/conveyal/r5/releases.
As of the time of writing, the latest version is version 7.2, *r5-v7.2-all.jar*.  
Download this and store in a local directory.

You will then need to setup the r5py config file, which among other things will 
point r5py to this file. On a windows machine *r5py* looks for this file in 
the `%APPDATA% \Roaming` directory. You can find this by typing 
`%APPDATA%` in the URL of an explorer window. Create a text file called 
`r5py.yml` in this directiory. The snippet below shows  the contents of this 
file on the testing computer.

```yaml
--max-memory=8G
--r5-classpath=C:\MyPrograms\r5\r5-v7.2-r5py-all.jar
--verbose=True
```

More information on r5py config files can be found on their documentation:
https://r5py.readthedocs.io/en/stable/user-guide/user-manual/configuration.html#configuration-via-config-files

### 7. Install Docker Desktop (only if using Valhalla)
Docker Desktop is available [here](https://www.docker.com/products/docker-desktop/)

> **_NOTE:_**  Docker Desktop installation requires admin privileges.




### 8. Download elevation data (only if using Valhalla)

Complete me!

Here's the link to a file, but add description
https://elevation-tiles-prod.s3.us-east-1.amazonaws.com/skadi/N44/N44W078.hgt.gz

## Running *accessopp*

### Open in Jupyter Lab

1. From the Windows Start Menu, open *Anaconda Prompt (Miniconda3)
2. Activate the `r5py` library by typing the following in the Miniconda prompt window
    ```console 
    (base) [local directory]> conda activate accessopp
    ```
3. Open a Jupyter Lab file, then open Jupyter lab
    - first change directory to where your notebooks are stored
        ```console 
        (accessopp) [local directory]> cd [notebook location] 
        (accessopp) [notebook location]> jupyter lab
        ```

### Run OpenTripPlanner



### Run *r5py*

**_NOTE:_** It may be necessary to specify the home location of Java. 
To find this path type `%APPDATA%` in the URL of an explorer window, navigate to 
`\AppData\Local\miniconda3\envs\r5py\Library\lib\jvm`. Copy this folder path, 
and add to the following code prior to running `accessopp`:

```console
(r5py) [local directory]> import os
(r5py) [local directory]> os.evniron["JAVA_HOME"] = r"[path from above]"

Several example scripts are included in the `example_notebooks` folder of this 
library package to help you get started.

Please also see the [**wiki**](https://github.com/bryce-sharman/accessopp/wiki) 
for this project for more detailed documentation and usage guides.




### Run Valhalla








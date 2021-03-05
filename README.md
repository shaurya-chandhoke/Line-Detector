## Author: Shaurya Chandhoke
### Line Detection

# Instructions for the User Running the Program
## Prerequisites
#### Creating a Python 3 Virtual Environment
It's essential the user running the program have the following set up:
- A terminal/shell (unix recommended)
- Python 3
- Python 3 pip
   - On Mac/Unix systems, installing Python 3 pip is as simple as `sudo apt-get install python3-pip`
   
After both are installed, it's **highly recommended** a python 3 virtual environment is created. To create one:

First install the `virtualenv` package via pip
```shell script
pip3 install virtualenv
```
After that's installed, create a python 3 virtual environment in the root directory of this project
```shell script
virtualenv venv
```
Once created, active the virtual environment
```shell script
source venv/bin/activate
```
If everything went smoothly, you should see a `(venv)` next to your terminal command line.

Now we can proceed with installing the prerequisite pip packages.

#### Installing the Prerequisite Packages
Included in the submission is a special *requirements.txt* file specially made for pip installations. In your terminal,
please run:
```shell script
pip3 install -r requirements.txt
```
It will install all the prerequisite python packages needed to run the program. You may open the file to view them.

#### NEW STEP PLEASE FOLLOW
Matplotlib is another essential dependency, however you may notice it's not part of the 
requirements.txt file. This is due to the matplotlib pip package setting a mandatory numpy
dependency requirement prior to installing, making it impossible to install it alongside numpy.
This is a known bug and is being worked on by the matplotlib team.

Now that the requirements.txt has been installed, you should now have all dependencies met to
install matplotlib. Making sure you're still in the (venv) environment, please run:

```shell
pip3 install matplotlib
```

It should install matplotlib. At this point, all dependencies for this project have been met.

## Running the Program
A quick way to run the program with its default configuration is:
```shell script
python3 line_detector.py <image file>
```

However, I've included a way to allow the user to fine tune the program.
To see all options available for the user:
```shell script
python3 line_detector.py --help
```

Below is a sample on how a user might run the program against an image:
```shell script
python3 line_detector.py ./img/road.png --gaussian-blur 2 --hessian-threshold 0.4
```

Sometimes, you may not want to view the output, but simply save:
```shell script
python3 line_detector.py ./img/road.png --hessian-threshold 0.3 --quiet 
```

Other times, you may not want to save, but simply view the output:
```shell script
python3 line_detector.py ./img/road.png --gaussian-blur 2 --hessian-threshold 0.2 --ransac-iteration 200 --nosave
```

If you do not pass the `--nosave` flag, all images will be saved in the **./out**
directory

## The Folder Structure of the Project
Contained within the submission should be a series of files and folders:

- /img
   - A series of sample images to test the program from. You may use your own if you like as long as you include the 
   correct path to the image. The program will let you know if it cannot find the image file.
- /out
   - A directory the program utilizes to write it's output images to. Please do not delete this directory.
- /src
   - A directory containing the source code for the image processing. I placed the code in this directory for better 
   readability and segmentation.
- line_detector.py
   - The Python file that contains the main function that will start the program. It is the file that will be run.
- requirements.txt
   - A special pip compatible package installation file that makes installation of prerequisite packages more 
   streamlined.
- README.md
   - An instructional file meant to serve as a quick How-To for running the program.
- README.txt
   - The same instructional file as README.md but in a non-markdown format in case a markdown viewer/text editor is not
   available.
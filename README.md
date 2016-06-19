# bag-of-words-image-classifier
My final project for the Udacity Machine Learning course: an image classifier that implements incremental learning, allowing it to learn new examples without having to retrain from scratch every time!

## Requirements

- Python 2.7
- Scikit-learn
- Pandas
- Numpy
- OpenCV 2.4x (NOT 3.x)

## Installing OpenCV 2.4x

Unlike the rest of the libraries, this one cannot be installed using pip or easy_install.  Use the following instructions depending on your OS.  Personally, I've successfully installed OpenCV on both Windows and Mac using these instructions.  Also, it's not been tested with OpenCV 3.x although it may still work.

### Windows

Download the binary [here](http://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html) and following [these instructions](http://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html#gsc.tab=0).  Note, if you use Anaconda, you'll need to copy the cv2.pyd file to C:\Anaconda2\Lib\site-packages .

### Mac

Make sure you have Homebrew installed and follow the instructions [here](https://jjyap.wordpress.com/2014/05/24/installing-opencv-2-4-9-on-mac-osx-with-python-support) (which includes where you can get Homebrew.)

Note: in step 2 of "Setting up Python", change 2.4.9 in the path to whatever the latest version of OpenCV 2.4x is (2.4.13 as of my writing now.)

### Linux

See [here](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)

Note: I also run Linux Mint 17.3 which has OpenCV preinstalled with Python support.  If you cannot install OpenCV (regardless of your OS), then I suggest installing [Linux Mint](https://www.linuxmint.com/download.php) under [Virtualbox](https://www.virtualbox.org/wiki/Downloads) as a last resort.  You may also need to install some of the above Python packages in this virtual Linux Mint installation.

## Setup

```
git clone https://github.com/xjdeng/bag-of-words-image-classifier.git
```

## Demonstration

Please open the IPython Notebook, RunMe.ipynb, to get an overview of how this program works, including the parameters, included libraries, etc.  To open an IPython Notebook from a command line, run the following after switching to the main directory of this program.

```
jupyter notebook
```

## Further documentation

Please see Writeup.pdf



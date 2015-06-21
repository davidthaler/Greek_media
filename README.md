# Greek Media Multilabel Classification (WISE 2014)
This repo holds the code for the 10th place entry in the 
2014 WISE/Greek Media Multi-label Classification competition hosted on Kaggle.
The competition page is [here](https://www.kaggle.com/c/wise-2014).   
  
## Description


## Requirements
* Python - this code was developed with Python 2.7
* Numpy/Scipy
* [Pandas](http://pandas.pydata.org/)
* [Scikit-Learn](http://scikit-learn.org/stable/index.html)
All of the dependencies are available in the [Anaconda Python distribution]
(https://store.continuum.io/cshop/anaconda/) for scientific computing.
* [Greek Media Data](https://www.kaggle.com/c/wise-2014/data) The files needed are: 
  * wise2014-libsvm-test
  * wise2014-libsvm-train
  * sampleSubmission

The arff files have the same data as the libsvm files, but just in a different format.
You will need to accept a set of competition rules (if you were not in the competition) before 
downloading.

## Set-up
The code assumes a directory structure under the project level of:  

* data/ - with the data from Kaggle  
* code/ - The name doesn't matter. This is where you clone to. 
* submissions/ - the code will write submissions here 

## Usage
The data comes in libsvm format. The first step, which happens once, is
to run rewriteTrain() and rewriteTest() from util.py to convert to array datatypes.

After that, a simple model that scores 0.769 on the final leaderboard can be generated as follows:



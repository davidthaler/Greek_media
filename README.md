# Kaggle-Greek Media Multilabel Classification (WISE 2014)
This repository holds the code for the 10th place entry in the 
Greek Media Multi-label Classification competition hosted on Kaggle.
The competition page is [here](https://www.kaggle.com/c/wise-2014).   
  
## Description
This task involves predicting topics or categories for each article 
in a corpus of about 100,000 Greek language news articles. 
Each document belongs to at least one category, but can belong to several. 

### Data
This is a multilabel-classification task. 
Each document belongs to at least one category, 
and about 1/3 of the documents are in multiple categories. 
There are 203 possible topics in all. 
The data are provided as row-normalized tf-idf features. 
The original text is not given. There are about 300k features.
There are 99780 total articles, of which the first 64857 form the training set,
while the remaining 34923 form the test set. 


### Evaluation Metric
The evaluation metric for this task is mean F1-score, 
which is explained [here](https://www.kaggle.com/c/wise-2014/details/evaluation). 
This is a per-example, or row-wise, F1-score.

### Models
The models included here are:
* UniformOVA - This model fits a linear support vector classifier to each class, 
in a one-versus-all style. Classification is by a global threshold on the decision values, 
plus a per-row threshold that is relative to the largest decision value for that instance.
This model is used in the runModel.py script and it scores 0.769 on the final leaderboard.
* ThresholdOVA - This is a variant of the UniformOVA model that adjusts the global threshold
on a per-class basis, so that there is one per-class threshold and one per-instance threshold.
It scores 0.771 on the final leaderboard.
* RidgePCA - This model performs PCA on the label matrix, to try to resolve the 203
labels into a smaller number of topics, then fits a ridge regression model to the retained
principal components. Classification is by thresholding on the predictions. 
This model scores 0.760 by itself, and it is used as a component of the StackModel().
* StackModel - This model runs several models inside of a cross-validation loop, aggregates
the hold-out set predictions and then computes features for a gradient boosted 
decision tree model from the aggregated predictions. 
The initial models used are: UniformOVA, RidgePCA and a ridge regression model 
trained on the count of positive labels for an instance.
This is the actual 10th place entry. It scores 0.7745 on the final leaderboard.


## Requirements
* Python - this code was developed with Python 2.7
* Numpy/Scipy
* [Pandas](http://pandas.pydata.org/)
* [Scikit-Learn](http://scikit-learn.org/stable/index.html) - 
All of the dependencies are available in the [Anaconda Python distribution]
(https://store.continuum.io/cshop/anaconda/) for scientific computing.
* [Greek Media Data](https://www.kaggle.com/c/wise-2014/data) - The files needed are: 
  * wise2014-libsvm-test
  * wise2014-libsvm-train
  * sampleSubmission

The arff files have the same data as the libsvm files, but in a different format.
You will need to accept a set of competition rules (if you were not in the competition) before 
downloading.

## Set-up
The code assumes a directory structure under the project level of:  

* data/ - with the data from Kaggle  
* code/ - The name doesn't matter. This is where you clone to. 
* submissions/ - the code will write submissions here 

## Usage
The string constant `BASE` in util.py contains the path to the project level directory.
You will need to edit that to point to where you put this project.   

The data comes in libsvm format. The first step, which happens once, 
is to convert to array datatypes by running the rewriteData.py script.   

After that, a simple model that scores 0.769 on the final leaderboard 
can be generated by running the runOne.py script. 
This script runs the UniformOVA model by itself. 
It should get 0.769 on the final standings (also good for 10th) 
and it runs fairly quickly.   

The actual 10th place entry was an ensemble generated with models.StackModel. 
You can run that model instead by uncommenting the line (in runModel.py):
```
model = StackModel(...   
```   
and it should score 0.775. It will take a couple of hours to run.

Note that for all models, the writeSubmission() function is fairly slow because it
has to iterate over the rows of a large label matrix and collapse it into a list-of
-lists format.



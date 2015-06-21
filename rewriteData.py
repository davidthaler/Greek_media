"""
This script transforms the libsvm-format data from Kaggle into numpy/scipy arrays.
It should be run once, before running any of the other code.

project: Kaggle WISE 2014 Greek Media competition
author: David Thaler
"""

import util
print 'transforming training data...'
rewriteTrain()
print 'transoforming test data...'
rewriteTest()
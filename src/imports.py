# for loading data script
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from scipy import io

# for the generation of the folds
from sklearn import model_selection

# for neural networks
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
from keras.models import Sequential
from keras.metrics import categorical_accuracy
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# to use in RF, SVM and QDA
from sklearn import metrics, multiclass
# Support Vector Machines
from sklearn import svm
# Random Forest
from sklearn import ensemble
# QDA
from sklearn import discriminant_analysis

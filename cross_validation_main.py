#essentials
import numpy as np
import pandas as pd
import os

##############################

print('loading libraries and modules used...')
execfile('imports.py')
print('end loading libraries and modules...')

##############################

print('loading data to analyse...')
execfile('load_data.py')
print('done loading data...')

##############################

print('Starting neural networks...')
execfile('results_nn.py')
print('neural networks end...')

##############################

print('Starting RF OVR...')
execfile('rfr_ovr.py')
print('RF OVR end...')

##############################

print('Starting RF multiclass...')
execfile('rfr_multi.py')
print('RF multiclass end...')

##############################

print('Starting SVM OVR...')
execfile('svm_ovr.py')
print('SVM OVR end...')

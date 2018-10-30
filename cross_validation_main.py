##############################
print('loading libraries and modules used...')
execfile('src/imports.py')
print('end loading libraries and modules...')

##############################

print('loading data to analyse...')
execfile('src/load_data.py')
print('done loading data...')

##############################

print('Starting neural networks...')
execfile('src/results_nn.py')
print('neural networks ended...')

##############################

print('Starting RF OVR...')
execfile('src/rf_ovr.py')
print('RF OVR ended...')

##############################

print('Starting RF multiclass...')
execfile('src/rf_multi.py')
print('RF multiclass ended...')

##############################

print('Starting SVM OVR linear kernel...')
execfile('src/svm_ovr_linear.py')
print('SVM OVR linear kernel ended...')

##############################

print('Starting SVM OVR rbf kernel...')
execfile('src/svm_ovr_kernel.py')
print('SVM OVR rbf kernel ended...')

##############################
print('Starting QDA...')
execfile('src/qda.py')
print('QDA ended...')

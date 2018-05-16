import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import io
import itertools

def plot_confusion_matrix(cm, cm_std, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, **kwarg):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_std = cm_std.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    cm = cm*100
    cm_std = cm_std*100
    
    cm = np.round(cm, decimals = 2)
    cm_std = np.round(cm_std, decimals = 2)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=20)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, weight='bold')
    plt.yticks(tick_marks, classes, weight='bold')

    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",size = 5, fontdict={'weight':'bold'},
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=15)
    plt.xlabel('Predicted label', size=15)
    

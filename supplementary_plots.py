#####################################################################################
######### Code to generate the plots in the supplementary material section ##########
#####################################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pylab as plt

#FIGURE S1: NN

list_hyparams = []

list_models=[[268,268,268,268], 
             [200,200,200,200],
             [100,100,100,100], 
             [50,50,50,50],
             [268,200,100,50],
             [268,128,64,32],
             [128,64,32,16], 
             [512,256,128,64],
             [250,250,250,250],
             [250,250,250],
             [250,250],
             [250],
             [500,500,500,500],
             [500,500,500],
             [500,500],
             [500],
             [125,125,125,125],
             [125,125,125],
             [125,125],
             [125]
             ]

list_hyparams.append(list_models)

list_cm=[]
acc_nn=[]

for i in range(20):
    acc_temp=[]
    cm_temp=[]
    
    for j in range(50):
        results=pd.read_csv('./results/task_tr_task_test_nn_models/categorical_'+str(i)+'_'+ str(j)+'_pred.csv')
        acc_temp.append(accuracy_score(results['true'].values,results['pred'].values))
        
    acc_nn.append(acc_temp)

acc_nn = np.asarray(acc_nn).reshape(20,5,10)

plt.style.use('default')
fig, ax = plt.subplots()
for i in range(acc_nn.shape[0]):
    
    temp=acc_nn[i,:,:].squeeze()
    
    i = i + 1
    y_to_plot=np.linspace(i-0.25, i+0.25, num=5)
    
    x_to_plot = 100*np.mean(temp, axis=1)
    err_to_plot = 100*np.std(temp, axis=1)
   
    ax.errorbar(y=y_to_plot, x=x_to_plot, xerr=err_to_plot, 
                c=sns.xkcd_rgb["dark salmon"],
                markersize=2,
                elinewidth=1,
                ecolor='g',
                fmt='o',
                capthick=2)
    ax.plot([np.mean(x_to_plot), np.mean(x_to_plot)], [i-0.25,i+0.25], 
             lw =3, color='k')

ax.set_yticks(range(1,21))
ax.set_yticklabels([str(elem) for elem in list_models], 
                   size='small',fontweight='bold')

ax.set_title('NEURAL NETWORKS',size = 20,fontweight='bold')
ax.set_ylabel('')
ax.set_xlabel('Accuracy (%)',size =15,fontweight='bold')
plt.savefig('./results/plots/supplementary/S1_fig.tiff',
            dpi=300,bbox_inches='tight')


#FIGURE S2: RF multi

folder_path='./results/rf_multi_grid/'

params = np.load(folder_path + 'param_grid.npy')
list_hyparams.append(params)

acc_rf_multi = []
for i in range(50):
    res=pd.read_csv(folder_path + 'res_fold_' + str(i)+'.csv')
    
    acc_rf_multi.append([accuracy_score(res.iloc[:,0], res.iloc[:,j]) for j in range(1,res.shape[1])])
    

acc_rf_multi = np.asarray(acc_rf_multi)
acc_rf_multi = acc_rf_multi.reshape(5,10,acc_rf_multi.shape[1])
acc_rf_multi = np.moveaxis(acc_rf_multi, -1, 0)

fig, axs = plt.subplots(ncols=2, sharey=True)
axs=axs.flatten()
for i in range(acc_rf_multi.shape[0]):
    
    temp=acc_rf_multi[i,:,:].squeeze()
    
    i = i + 1
    y_to_plot=np.linspace(i-0.25, i+0.25, num=5)
    
    x_to_plot = 100*np.mean(temp, axis=1)
    err_to_plot = 100*np.std(temp, axis=1)
    #print(x_to_plot)
    #ax.scatter(x=x_to_plot, y=100*y_to_plot, s=4, c=sns.xkcd_rgb["dark salmon"])
    axs[0].errorbar(y=y_to_plot, x=x_to_plot, xerr=err_to_plot, 
                c=sns.xkcd_rgb["dark salmon"],
                markersize=2,
                elinewidth=1,
                ecolor='g',
                fmt='o',
                capthick=2)
    axs[0].plot([np.mean(x_to_plot), np.mean(x_to_plot)], [i-0.25,i+0.25], 
             lw =3, color='k')

axs[0].set_yticks(range(1,acc_rf_multi.shape[0]+1))
axs[0].set_yticklabels([str(param).replace("estimators", "trees") for param in params], 
                   size='small',fontweight='bold')

axs[0].set_title('RF MULTI',size = 20,fontweight='bold')
axs[0].set_ylabel('')
axs[0].set_xlabel('Accuracy (%)',size =15,fontweight='bold')

#plt.savefig('./results/plots/supplementary/accuracies_tfmri_rf_multi.png',
 #           dpi=600,bbox_inches='tight')


#FIGURE S3: RF ovr

folder_path='./results/rf_ovr_grid/'

params = np.load(folder_path + 'param_grid.npy')
list_hyparams.append(params)

acc_rf_ovr = []
for i in range(50):
    res=pd.read_csv(folder_path + 'res_fold_' + str(i)+'.csv')
    
    acc_rf_ovr.append([accuracy_score(res.iloc[:,0], res.iloc[:,j]) for j in range(1,res.shape[1])])
    

acc_rf_ovr = np.asarray(acc_rf_ovr)
acc_rf_ovr = acc_rf_ovr.reshape(5,10,acc_rf_ovr.shape[1])
acc_rf_ovr = np.moveaxis(acc_rf_ovr, -1, 0)

for i in range(acc_rf_ovr.shape[0]):
    
    temp=acc_rf_ovr[i,:,:].squeeze()
    
    i = i + 1
    y_to_plot=np.linspace(i-0.25, i+0.25, num=5)
    
    x_to_plot = 100*np.mean(temp, axis=1)
    err_to_plot = 100*np.std(temp, axis=1)

    axs[1].errorbar(y=y_to_plot, x=x_to_plot, xerr=err_to_plot, 
                c=sns.xkcd_rgb["dark salmon"],
                markersize=2,
                elinewidth=1,
                ecolor='g',
                fmt='o',
                capthick=2)
    axs[1].plot([np.mean(x_to_plot), np.mean(x_to_plot)], [i-0.25,i+0.25], 
             lw =3, color='k')

axs[1].set_title('RF OVR ',size = 20,fontweight='bold')
axs[1].set_ylabel('')
axs[1].set_xlabel('Accuracy (%)',size =15,fontweight='bold')
plt.savefig('./results/plots/supplementary/S2_fig.tiff',
            dpi=300,bbox_inches='tight')

#FIGURE S4: SVM Linear

folder_path='./results/svm_ovr_linear/'

params = np.load(folder_path + 'param_grid.npy')
list_hyparams.append(params)

acc_svm_linear = []
for i in range(50):
    res=pd.read_csv(folder_path + 'res_fold_' + str(i)+'.csv')
    
    acc_svm_linear.append([accuracy_score(res.iloc[:,0], res.iloc[:,j]) for j in range(1,res.shape[1])])
    

acc_svm_linear = np.asarray(acc_svm_linear)
acc_svm_linear = acc_svm_linear.reshape(5, 10, acc_svm_linear.shape[1])
acc_svm_linear = np.moveaxis(acc_svm_linear, -1, 0)

fig, ax = plt.subplots()
for i in range(acc_svm_linear.shape[0]):
    
    temp=acc_svm_linear[i,:,:].squeeze()
    
    i = i + 1
    y_to_plot=np.linspace(i-0.25, i+0.25, num=5)
    
    x_to_plot = 100*np.mean(temp, axis=1)
    err_to_plot = 100*np.std(temp, axis=1)

    ax.errorbar(y=y_to_plot, x=x_to_plot, xerr=err_to_plot, 
                c=sns.xkcd_rgb["dark salmon"],
                markersize=2,
                elinewidth=1,
                ecolor='g',
                fmt='o',
                capthick=2)
    ax.plot([np.mean(x_to_plot), np.mean(x_to_plot)], [i-0.25,i+0.25], 
             lw =3, color='k')

ax.set_yticks(range(1,acc_svm_linear.shape[0]+1))
ax.set_yticklabels([str(param).replace('alpha','C') for param in params], 
                   size='small',fontweight='bold')

ax.set_title('SVM LINEAR KERNEAL' ,size = 20,fontweight='bold')
ax.set_ylabel('')
ax.set_xlabel('Accuracy (%)',size =15,fontweight='bold')

plt.savefig('./results/plots/supplementary/S3_fig.tiff',
            dpi=300,bbox_inches='tight')


#FIGURE S4: SVM KERNEL

folder_path='./results/svm_ovr_kernel/'

params = np.load(folder_path + 'param_grid.npy')
list_hyparams.append(params)

acc_svm_kernel = []
for i in range(50):
    res=pd.read_csv(folder_path + 'res_fold_' + str(i)+'.csv')
    
    acc_svm_kernel.append([accuracy_score(res.iloc[:,0], res.iloc[:,j]) for j in range(1,res.shape[1])])
    

acc_svm_kernel = np.asarray(acc_svm_kernel)
acc_svm_kernel = acc_svm_kernel.reshape(5,10,acc_svm_kernel.shape[1])
acc_svm_kernel = np.moveaxis(acc_svm_kernel, -1, 0)

fig, ax = plt.subplots()
for i in range(acc_svm_kernel.shape[0]):
    
    temp=acc_svm_kernel[i,:,:].squeeze()
    
    i = i + 1
    y_to_plot=np.linspace(i-0.25, i+0.25, num=5)
    
    x_to_plot = 100*np.mean(temp, axis=1)
    err_to_plot = 100*np.std(temp, axis=1)
   
    ax.errorbar(y=y_to_plot, x=x_to_plot, xerr=err_to_plot, 
                c=sns.xkcd_rgb["dark salmon"],
                markersize=2,
                elinewidth=1,
                ecolor='g',
                fmt='o',
                capthick=2)
    ax.plot([np.mean(x_to_plot), np.mean(x_to_plot)], [i-0.25,i+0.25], 
             lw =3, color='k')

ax.set_yticks(range(1,acc_svm_kernel.shape[0]+1))
ax.set_yticklabels([str(param).replace('alpha','C').replace('gamma', '$\gamma$') for param in params], 
                   size='small',fontweight='bold')

ax.set_title('SVM RBF KERNEL' ,size = 20,fontweight='bold')
ax.set_ylabel('')
ax.set_xlabel('Accuracy (%)',size =15,fontweight='bold')

plt.savefig('./results/plots/supplementary/S4_fig.tiff',
            dpi=300,bbox_inches='tight')

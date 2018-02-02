import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import io
from confusion_matrix import plot_confusion_matrix
from sklearn import metrics
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.patches as mpatches
from nilearn import plotting, image

from scipy import interp
from sklearn.preprocessing import label_binarize
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

class_names = [' VIS', ' SM', 'DA', 'VA','L', 'FP','DMN', 'SUB', 'CER']
colors = ['dark blue','tan','lavender','olive','peach','puke green','blue grey', 'black','brown']

folder_path = './results/task_tr_task_test_nn_models/'
acc =[]
rec =[]
prec = []

for i in xrange(len(list_models)):
    name_model='categorical_' + str(i)
    acc_m = []
    rec_m =[]
    prec_m = []    
    for iter_id in xrange(5):    
        res = pd.read_csv(folder_path + name_model+str(iter_id)+'_pred.csv')    
        acc_m.append(sum(res['pred'].values==res['true'].values)/float(res.shape[0]))
        rec_m.append(metrics.recall_score(res['true'].values,res['pred'].values, average=None))
        prec_m.append(metrics.precision_score(res['true'].values,res['pred'].values, average=None))
        
    acc.append(np.asarray(acc_m))
    rec.append(np.asarray(rec_m))
    prec.append(np.asarray(prec_m))
    
acc = np.asarray(acc)
rec = np.asarray(rec)
prec = np.asarray(prec)

for i in xrange(len(list_models)):
    print('accuracy of model: ', list_models[i], ' = ', np.mean(acc[i]))
    print('recall of model: ', list_models[i], ' = ', np.mean(rec[i], axis=1))
    print('prec of model: ', list_models[i], ' = ', np.mean(prec[i],axis=1))
    print(" ")


x_to_plot = np.vstack([np.repeat(i,5) for i in range(1,21)])
acc_mean = np.mean(acc, axis=1)

#FIGURE 1
fig, ax = plt.subplots()
sns.swarmplot(x=x_to_plot.flatten(), y=100*acc.flatten(), color= sns.xkcd_rgb["dark salmon"], ax=ax)
for tick in ax.get_xticks():
    ax.plot([tick-0.25, tick +0.25], [100*acc_mean[tick],100*acc_mean[tick]], lw =3, color='k')
ax.set_xticks(range(0,21))
ax.set_xticklabels(list_models,rotation=90)
ax.set_title('Motor Task 5-Cross Validation',size = 'xx-large')
ax.set_xlabel('Neural Network Models', position = (0.5,-0.8), size = 'large', labelpad = 20)
ax.set_ylabel('Accuracy + SD (%)',size = 'large')
fig.savefig('plots/accuracies_tfmri_models.png',dpi=300,bbox_inches='tight')


#FIGURE 2
#We saw that the best case is [500]. We shall focus then on this case (file with 15)
name_model = 'categorical_' + str(15)
cfm_list =[]
for iter_id in xrange(5):    
        res = pd.read_csv(folder_path + name_model+str(iter_id)+'_pred.csv')    
        cm =metrics.confusion_matrix(res['true'], res['pred'])
        cfm_list.append(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis])

cfm_list = np.asarray(cfm_list)
cm = np.mean(cfm_list, axis=0)
cm_std = np.std(cfm_list, axis = 0)
np.set_printoptions(precision = 2)

plt.figure()
plot_confusion_matrix(cm, cm_std, classes=class_names, normalize=False,
                      title='Task-Training Task-Test (%)', cmap = plt.cm.YlGn) #plt.cm.Greys for resting
plt.savefig('plots/confusion_task_train_task_test.png',dpi=300,bbox_inches='tight')


#FIGURE 3
res = pd.read_csv('results/task_tr_rest_tr/task_tr_resting_ts_pred.csv')
cm =metrics.confusion_matrix(res['true'], res['pred'])
cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

plt.figure()
plot_confusion_matrix(cm, np.zeros((9,9)), classes=class_names, normalize=False,
                      title='Task-Training Resting-Test (%)', cmap = plt.cm.YlGn)
plt.savefig('plots/confusion_task_train_rest_test.png',dpi=300,bbox_inches='tight')



#FIGURE 4
yeoROIs=io.loadmat('data/Shen268_yeo_RS7.mat',squeeze_me=True)['yeoROIs']-1
rsn_ROIs= [class_names[i] for i in yeoROIs]

array_dirs = sorted([f for f in os.listdir('./data/data_task_icafix/') if f.startswith('sub-')])
corrs_list = [np.corrcoef(np.loadtxt('./data/data_task_icafix/' + f + '/func_mean.txt')) for f in array_dirs]
corrs_list = [mat - np.identity(mat.shape[0]) for mat in corrs_list]
corr_task = np.asarray(corrs_list)
corr_task = corr_task.mean(axis=0)

task_corr = pd.DataFrame(corr_task[np.argsort(yeoROIs),:].transpose()).corr().values
                        
fig= plt.figure(figsize=(10,5))
ax1=plt.subplot(1,2,1)
hm = sns.heatmap(task_corr,square=True,cbar_kws={"shrink": .6},
                 xticklabels=False,yticklabels=False, cmap=plt.cm.RdBu_r, ax=ax1)
x=0
y=0
w = [sum(yeoROIs==i) for i in range(9)]
for i in range(9):

    hm.add_patch(mpatches.Rectangle((x,y), w[i], w[i], fill=False,edgecolor='k',lw=3))
    x= x + w[i]
    y = y+w[i]
dat_to_plot=[]
for i, name in enumerate(class_names):
    rsn_inds = np.where(np.sort(yeoROIs)==i)
    mat = task_corr[np.meshgrid(rsn_inds[0],rsn_inds[0], indexing ='ij')]
    #take upper off diagonal terms
    dat_to_plot.append(mat[np.triu_indices(mat.shape[0], k=1)])


ax2=plt.subplot(1, 2, 2)
x_for_boxplot = [np.repeat(i+1, len(dat_to_plot[i])) for i in range(9)]
x_for_boxplot = np.hstack(x_for_boxplot)

y_for_boxplot = np.hstack(dat_to_plot)

bp= sns.boxplot(x =x_for_boxplot, y=y_for_boxplot, linewidth=2.5, 
                whis = 1.5,showfliers=False, width=0.6,ax=ax2)#, ax=ax[1])
for i,box in enumerate(bp.artists):
    box.set_edgecolor('k')
    # iterate over whiskers and median lines
    for j in range(5*i,5*(i+1)):
         bp.lines[j].set_color('k')
         
    bp.lines[5*i+4].set_color(sns.xkcd_rgb['off white'])
for patch, color in zip(bp.artists, sns.xkcd_palette(colors)):
    patch.set_facecolor(color)
sns.swarmplot(x =x_for_boxplot, y=y_for_boxplot, size=1.2, color='0.3', ax=ax2)#, ax=ax[1])
bp.set_xticklabels(class_names)
ax2.set_ylim(-0.8,1)
plt.xticks(rotation=90)
plt.ylabel("Pearson similarity", size=20)
fig.subplots_adjust(wspace=0.3)
ax2.set_aspect(4)
plt.rcParams["font.weight"] = "bold"
plt.savefig('plots/cors_mean_patterns.png', dpi=300,bbox_inches='tight')


#FIGURE 5
plt.rcParams.update(plt.rcParamsDefault)
yt = res['true']
n_classes= len(np.unique(yt))
y_test = label_binarize(yt.values,np.unique(yt.values))

y_score = np.load('results/task_tr_rest_tr/task_tr_resting_ts_pred_probs.npy')

fpr = dict()
tpr = dict()
thr = dict()
roc_auc = dict()
for i in range(9):
    fpr[i], tpr[i], thr[i] = metrics.roc_curve(y_test[:,i], y_score[:, i])
    print(metrics.roc_auc_score(y_test[:,i], y_score[:, i],average="micro"))
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], thr = metrics.roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = ['dark blue','tan','lavender','olive','peach','puke green','blue grey', 'black','brown']

fpr_model = [sum(np.argmax(y_score, axis=1)[yt.values!=i]==i)/float(sum(yt.values!=i)) for i in range(n_classes)]
rec_model = list(metrics.recall_score(yt.values, np.argmax(y_score, axis=1),average=None))
prec_model = list(metrics.precision_score(yt.values, np.argmax(y_score, axis=1),average=None))

fig, ax = plt.subplots()
ax.scatter(fpr_model, rec_model, marker='x',c ='#363737',s=100,zorder=2)
for i, color in zip(range(n_classes), sns.xkcd_palette(colors)):
    ax.plot(fpr[i], tpr[i], color=color, lw=3,
             label='{0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]),zorder=1)
axins=fig.add_axes([0.45, 0.3, 0.4, 0.4])

axins.scatter(fpr_model, rec_model, marker='x', c ='#363737',s=100, zorder=2)
for i, color in zip(range(n_classes), sns.xkcd_palette(colors)):
    axins.plot(fpr[i], tpr[i], color=color, lw=3, zorder=1)
            # label='{0} (area = {1:0.2f})'
             #''.format(class_names[i], roc_auc[i]))
axins.set_xlim([0.,0.15])
axins.set_ylim([0.4,1.0])
mark_inset(ax, axins, loc1=1, loc2=3, fc="none",lw=2, ec = 'k', ls = 'dashed')
ax.legend(bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0.)
ax.set_title('ROC curve', fontsize = 20)
ax.set_xlabel('False Positive Rate', fontsize = 15)
ax.set_ylabel('True Positive Rate', fontsize = 15)
ax.set_xlim([-0.1,1])
ax.set_ylim([-0.1,1.1])
plt.savefig('plots/ROC_curves.png', dpi=300,bbox_inches='tight')

#FIGURE 6
precision = dict()
recall = dict()
thresholds = dict()
average_precision = dict()

for i in range(9):
    precision[i], recall[i], thresholds[i] = metrics.precision_recall_curve(y_test[:,i], y_score[:,i])
    average_precision[i] = metrics.average_precision_score(y_test[:, i], y_score[:, i])

rec_model = list(metrics.recall_score(yt.values, np.argmax(y_score, axis=1),average=None))
prec_model = list(metrics.precision_score(yt.values, np.argmax(y_score, axis=1),average=None))

fig, ax = plt.subplots()

for i, color in zip(range(n_classes), sns.xkcd_palette(colors)):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2, zorder=1,
                  label= '{0} (area = {1:0.2f})'
                 ''.format(class_names[i], average_precision[i]))
plt.scatter(rec_model, prec_model,marker='x', c ="#363737", s=100,zorder=2)
plt.legend(loc="lower left", ncol =2)
plt.title('PR curve', fontsize = 20)
plt.xlabel('Recall', fontsize = 15)
plt.ylabel('Precision', fontsize = 15)
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.savefig('plots/PR_curves.png', dpi=300,bbox_inches='tight')


#FIGURE 7
rsn_labels=io.loadmat('data/Shen268_yeo_RS7.mat')['yeoROIs'].flatten()
shuf_ind=np.load('results/task_tr_rest_tr/shuffl_ind_rest_test.npy')
nodes_id = np.tile(np.arange(1,269), 282)[shuf_ind]
nodes_acc=np.array([sum(res['pred'][nodes_id==idx]==res['true'][nodes_id==idx])/282.0 for idx in np.arange(1,269)])

vmax = max(nodes_acc)
vmin = min(nodes_acc)
for i in range(1,10):
    
    #we use 1mm resolution for a finner result
    img =  image.load_img("./data/atlas/shen_1mm_268_parcellation.nii.gz")
    
    a = img.get_data()
    sub = np.where(rsn_labels==i)[0] + 1
    
    
    for index in np.ndindex(a.shape[0],a.shape[1],a.shape[2]):
        xx=index[0]
        yy=index[1]
        zz=index[2]
        
        if a[xx,yy,zz] in sub:
            a[xx,yy,zz] = nodes_acc[a[xx,yy,zz]-1]
        else:
            a[xx,yy,zz] = 0
             
        
    tmap = image.new_img_like(img,a, copy_header =True)
    display = plotting.plot_glass_brain(tmap, threshold=vmin,display_mode='lzr',
                              title = class_names[i-1],
                              vmax=vmax,black_bg=False,
                              colorbar=True,symmetric_cbar=False)
    display.savefig("./plots/" + class_names[i-1]+"_glass_brain.png", dpi=300)
    display.close()

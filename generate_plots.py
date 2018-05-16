#############################################
### Load needed packages and libraries#######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import io
from confusion_matrix import plot_confusion_matrix
from sklearn import metrics
import matplotlib.patches as mpatches
from nilearn import plotting, image

from scipy import interp
from sklearn.preprocessing import label_binarize

import os
if not os.path.exists('./plots'):
    os.mkdir('./plots')

class_names = [' VIS', ' SM', 'DA', 'VA','L', 'FP','DMN', 'SUB', 'CER']
n_classes= len(class_names)

colors = ['dark blue','tan','lavender','olive','peach','puke green','blue grey', 'black','brown']

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



folder_path = './results/task_tr_task_test_nn_models/'
list_acc=[]
for i in range(20):
    acc_temp=[]
    
    for j in range(50):
        results=pd.read_csv(folder_path + 'categorical_'+str(i)+'_'+ str(j)+'_pred.csv')
        acc_temp.append(metrics.accuracy_score(results['true'].values,results['pred'].values))

    list_acc.append(acc_temp)



folder_path='./results/qda/'
acc_m = []
for i in range(50):
    res=pd.read_csv(folder_path + 'res_fold_' + str(i)+'.csv')
    acc_m.append(metrics.accuracy_score(res['true'].values,res['pred'].values))

list_acc.append(acc_m)


folder_path='./results/rf_multi/'
acc_m = []
for i in range(50):
    res=pd.read_csv(folder_path + 'res_fold_' + str(i)+'.csv')
    acc_m.append(metrics.accuracy_score(res['true'].values,res['pred'].values))
    
list_acc.append(acc_m)

folder_path='./results/rf_ovr/'
acc_m = []
for i in range(50):
    res=pd.read_csv(folder_path + 'res_fold_' + str(i)+'.csv')
    acc_m.append(metrics.accuracy_score(res['true'].values,res['pred'].values))
    
list_acc.append(acc_m)

folder_path='./results/svm_ovr/'
acc_m = []
for i in range(50):
    res=pd.read_csv(folder_path + 'res_fold_' + str(i)+'.csv')
    acc_m.append(metrics.accuracy_score(res['true'].values,res['pred'].values))
    
list_acc.append(acc_m)

list_xticks=[]

for i in range(20):
    list_xticks.append('NN:'+str(list_models[i]))

list_xticks.append('QDA')
list_xticks.append('RF (Multi)')
list_xticks.append('RF (OVR)')
list_xticks.append('SVM (OVR)')

#FIGURE 2
fig, ax = plt.subplots()
for i in range(4,100,4):
    temp=np.array(list_acc[int(i/4.0-1)])
    y_to_plot=[i-1,i-0.5,i,i+0.5,i+1]
    
    x_to_plot = [100*np.mean(temp[range(5*j, 5*(j+1))]) for j in range(5)]
    err_to_plot = [100*np.std(temp[range(5*j, 5*(j+1))]) for j in range(5)]

    ax.errorbar(y=y_to_plot, x=x_to_plot, xerr=err_to_plot, 
                c=sns.xkcd_rgb["dark salmon"],
                markersize=2,
                elinewidth=1,
                ecolor='g',
                fmt='o',
                capthick=2)
    ax.plot([np.mean(x_to_plot), np.mean(x_to_plot)], [i-1,i+1], lw =3, color='k')
        
ax.set_yticks(range(4,100,4))
ax.set_xticks(range(54,84,2))
ax.set_xticklabels(range(54,84,2),fontweight='bold')
ax.set_yticklabels(list_xticks, size='small',fontweight='bold')
ax.set_title('Motor Task  \n' + '5-Repeated 10-Fold CV',size = 20,fontweight='bold')
ax.set_ylabel('')
ax.set_xlabel('Accuracy (%)',size =15,fontweight='bold')
plt.savefig('plots/accuracies_tfmri_models.png',dpi=300,bbox_inches='tight')
plt.savefig('plots/accuracies_tfmri_models.eps',dpi=300,bbox_inches='tight')


# FIGURE 3
# We saw that the best case is neural network [512,256,128,64]. We shall focus then on this case (file with 7)
folder_path = './results/task_tr_task_test_nn_models/'
name_model = 'categorical_' + str(7)
list_cm=[]
for i in range(50):
    results=pd.read_csv(folder_path + name_model + '_'+str(i)+'_pred.csv')
    
    cm = metrics.confusion_matrix(results['true'].values,results['pred'].values)    
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    
    list_cm.append(cm)


cm_mean=np.mean(list_cm, axis=0)
cm_std=np.std(list_cm, axis=0)

fig, ax = plt.subplots()
plot_confusion_matrix(cm_mean, cm_std, classes=class_names, normalize=False,
                      title='Task-Training \n Task-Test (%)', cmap = plt.cm.YlGn) #plt.cm.Greys f
plt.savefig('plots/confusion_task_train_task_test.png',dpi=300,bbox_inches='tight')
plt.savefig('plots/confusion_task_train_task_test.eps',dpi=300,bbox_inches='tight')


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
                whis = 1.5,showfliers=False, width=0.6,ax=ax2)
for i,box in enumerate(bp.artists):
    box.set_edgecolor('k')
    # iterate over whiskers and median lines
    for j in range(5*i,5*(i+1)):
         bp.lines[j].set_color('k')
         
    bp.lines[5*i+4].set_color(sns.xkcd_rgb['off white'])
for patch, color in zip(bp.artists, sns.xkcd_palette(colors)):
    patch.set_facecolor(color)
sns.swarmplot(x =x_for_boxplot, y=y_for_boxplot, size=1.2, color='0.3', ax=ax2)
bp.set_xticklabels(class_names)
ax2.set_ylim(-0.8,1)
plt.xticks(rotation=90)
plt.ylabel("Pearson similarity", size=20,fontweight='bold')
fig.subplots_adjust(wspace=0.3)
ax2.set_aspect(4)
#plt.rcParams["font.weight"] = "bold"
plt.savefig('plots/cors_mean_patterns.png', dpi=300,bbox_inches='tight')
plt.savefig('plots/cors_mean_patterns.eps', dpi=300,bbox_inches='tight')


#FIGURE 5
list_cm=[]
for i in range(50):
    results=pd.read_csv('./results/task_tr_rest_tr/task_tr_resting_ts_'+str(i)+'_pred.csv')
    
    cm = metrics.confusion_matrix(results['true'].values,results['pred'].values)    
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    
    list_cm.append(cm)


cm_mean=np.mean(list_cm, axis=0)
cm_std=np.std(list_cm, axis=0)

fig, ax = plt.subplots()
plot_confusion_matrix(cm_mean, np.zeros((9,9)), classes=class_names, normalize=False,
                      title='Task-Training \n Resting-Test (%)', cmap = plt.cm.YlGn)
plt.savefig('plots/confusion_task_train_rest_test.png',dpi=300,bbox_inches='tight')
plt.savefig('plots/confusion_task_train_rest_test.eps',dpi=300,bbox_inches='tight')



#FIGURE 6
list_scores=[]
list_res =[]
for i in range(50):
    results=pd.read_csv('./results/task_tr_rest_tr/task_tr_resting_ts_'+str(i)+'_pred.csv')
    list_res.append(results.loc[:,['pred','true']].values)
    scores=np.load('./results/task_tr_rest_tr/task_tr_resting_ts_'+str(i) +'_pred_probs.npy')
    list_scores.append(scores)
    

tpr_model= np.array([metrics.recall_score(list_res[i][:,1],list_res[i][:,0],average =None) \
                     for i in range(50)])

fpr_model= np.array([[sum(list_res[i][:,0][list_res[i][:,1]!= clas_id]==clas_id)/float(sum(list_res[i][:,1]!= clas_id)) \
                      for clas_id in range(n_classes)] for i in range(50)])

fig, ax = plt.subplots()

for i, color in zip(range(n_classes), sns.xkcd_palette(colors)):

    tprs=[]
    aucs=[]
    mean_fpr = np.linspace(0, 1, 1000)
    
    for fold in range(50):
        y_test=label_binarize(list_res[fold][:,1],[0,1,2,3,4,5,6,7,8,9])
        y_score=list_scores[fold]
    
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:,i], y_score[:,i])
        auc_roc = metrics.roc_auc_score(y_test[:, i], y_score[:, i])
        aucs.append(auc_roc)
        tprs.append(interp(mean_fpr, fpr, tpr))    
        
    mean_tpr=np.mean(tprs, axis=0)
    mean_auc= np.mean(aucs, axis=0)
    std_auc= np.std(aucs, axis=0)
    mean_tpr[-1] = 1.0
    l, = plt.plot(mean_fpr, mean_tpr, color=color, lw=2, zorder=1,
                  label= class_names[i] + ' (%0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',alpha=.3)

plt.scatter(np.mean(fpr_model,axis=0), np.mean(tpr_model,axis=0),marker='x', c ="#363737", s=100,zorder=2, 
            linewidths=100, edgecolors=None)

plt.legend(loc="lower right", ncol =1,
           prop={'weight':'bold'},labelspacing=0.1,columnspacing=0.2, frameon=False)
plt.title('ROC curve', fontsize = 20, fontweight='bold')
plt.xlabel('FALSE POSITIVE RATE', fontsize = 15,fontweight='bold')
plt.ylabel('TRUE POSITIVE RATE', fontsize = 15,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.savefig('./plots/ROC_curves.png', dpi=300,bbox_inches='tight')
plt.savefig('./plots/ROC_curves.eps', dpi=300,bbox_inches='tight')


#FIGURE 7

rec_model = np.array([metrics.recall_score(list_res[i][:,1],list_res[i][:,0],average =None) for i in range(50)])
prec_model = np.array([metrics.precision_score(list_res[i][:,1],list_res[i][:,0],average =None) for i in range(50)])

fig, ax = plt.subplots()
for i, color in zip(range(n_classes), sns.xkcd_palette(colors)):

    precs=[]
    avg=[]
    mean_rec = np.linspace(0, 1, 1000)
    
    for fold in range(50):
        y_test=label_binarize(list_res[fold][:,1],[0,1,2,3,4,5,6,7,8,9])
        y_score=list_scores[fold]
    
        precision, recall, thresholds = metrics.precision_recall_curve(y_test[:,i], y_score[:,i])
        average_precision = metrics.average_precision_score(y_test[:, i], y_score[:, i])
        avg.append(average_precision)
        precs.append(interp(mean_rec, recall[::-1], precision[::-1]))    
        
    mean_prec=np.mean(precs, axis=0)
    mean_prec[-1] = 0.0
    mean_avg= np.mean(avg, axis=0)
    std_avg= np.std(avg, axis=0)
    
    l, = plt.plot(mean_rec, mean_prec, color=color, lw=2, zorder=1,
                  label= class_names[i] + ' (%0.2f $\pm$ %0.2f)' % (mean_avg, std_avg))

plt.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r',alpha=.3)

plt.scatter(np.mean(rec_model,axis=0), np.mean(prec_model, axis=0),marker='x', c ="#363737", s=100,zorder=2, 
            linewidths=100, edgecolors=None)
            
plt.legend(loc='lower left', ncol =2, prop={'weight':'bold'},labelspacing=0.1,columnspacing=0.2, frameon=False)
plt.title('PR curve', fontsize = 20, fontweight='bold')
plt.xlabel('Recall', fontsize = 15,fontweight='bold')
plt.ylabel('Precision', fontsize = 15,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.savefig('./plots/PR_curves.png', dpi=300,bbox_inches='tight')
plt.savefig('./plots/PR_curves.eps', dpi=300,bbox_inches='tight')


#FIGURE 8
rsn_labels=io.loadmat('data/Shen268_yeo_RS7.mat')['yeoROIs'].flatten()

rois_ind_all=[np.load('./results/task_tr_rest_tr/task_tr_resting_ts_'+ str(fold)+'.npz')['rois_ids'] for fold in range(50)]
res_all=[pd.read_csv('./results/task_tr_rest_tr/task_tr_resting_ts_'+ str(fold)+'_pred.csv') for fold in range(50)]

nodes_acc=[np.asarray([metrics.accuracy_score(res_all[fold].iloc[rois_ind_all[fold]==idx,2],
                       res_all[fold].iloc[rois_ind_all[fold]==idx,1]) 
    for idx in np.arange(268)]) for fold in range(50)]

nodes_acc = np.vstack(nodes_acc)

nodes_acc = np.mean(nodes_acc, axis=0)

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
            a[xx,yy,zz] = nodes_acc[int(a[xx,yy,zz])-1]
        else:
            a[xx,yy,zz] = 0
             
        
    tmap = image.new_img_like(img,a, copy_header =True)
    if i==9:
        display = plotting.plot_glass_brain(tmap, threshold=vmin,display_mode='lzr',
                                  title = class_names[i-1],
                                  vmax=vmax,black_bg=False,
                                  colorbar=True,symmetric_cbar=False)
    else:
        display = plotting.plot_glass_brain(tmap, threshold=vmin,display_mode='lzr',
                                  title = class_names[i-1],
                                  vmax=vmax,black_bg=False,
                                  symmetric_cbar=False)
        
    display.savefig("./plots/" + class_names[i-1]+"_glass_brain.png", dpi=300)
    display.savefig("./plots/" + class_names[i-1]+"_glass_brain.eps", dpi=300)
    display.close()

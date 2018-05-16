# 1. Import libraries and modules
import numpy as np
import pandas as pd
from scipy import io
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.metrics import categorical_accuracy
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from sklearn import model_selection

# Select outfile name for data and model and define constants of the neural networks
out_file='task_tr_resting_ts_' 
batch_size = 128
epochs = 500
drop=0.2;
patience=20
loss='categorical_crossentropy'

# for reproducibility (Note that the seed is different than the one for model selection).
# this is done to have different partitions in the training  data
# We are not now comparing between models, just testing the generalisation to resting data
seed = 250 

fold_model = "./results/task_tr_rest_tr/"
if not os.path.exists(fold_model):
    os.makedirs(fold_model)

#we saw previously this was the best model
list_models=[[512, 256, 128, 64]]
to_do=range(len(list_models))


labels = io.loadmat('./data/Shen268_yeo_RS7.mat',variable_names = 'yeoROIs',squeeze_me =True)['yeoROIs']

####################### TRAIN DATA#############################################
#load time series, calculate the correlation matrices, substract 1 from diagonal and and randomise input data
print('load time series, calculate the correlation matrices and substract 1 from diagonal for TASK DATA...')

array_dirs = sorted([f for f in os.listdir('./data/data_task_icafix') if f.startswith('sub-')])

corrs_list = [np.corrcoef(np.loadtxt('data/data_task_icafix/' + f + '/func_mean.txt')) for f in array_dirs]
corrs_list = [mat - np.identity(mat.shape[0]) for mat in corrs_list]
XX_task = np.array(corrs_list)

#y_task = np.tile(labels, len(corrs_list))
#y_task =label_binarize(y_task,classes=[1,2,3,4,5,6,7,8,9])

####################### TEST DATA##############################################
#load time series, calculate the correlation matrices, substract 1 from diagonal and and randomise input data
print('load time series, calculate the correlation matrices and substract 1 from diagonal for RESTING DATA...')

array_dirs = sorted([f for f in os.listdir('./data/data_fmri_clean') if f.startswith('sub-')])

corrs_list = [np.corrcoef(np.loadtxt('./data/data_fmri_clean/' + f + '/func_mean.txt')) for f in array_dirs]
corrs_list = [mat - np.identity(mat.shape[0]) for mat in corrs_list]
XX_rest = np.array(corrs_list)

###############################################################
### create shufflings of 

index_subjects = np.arange(XX_rest.shape[0])

rkf=model_selection.RepeatedKFold(n_splits=10, n_repeats=5, random_state=seed)

train_index_list=[]
test_index_list=[]
for train_index,test_index in rkf.split(index_subjects):
    train_index_list.append(train_index)
    test_index_list.append(test_index)


n_subjects=(XX_rest.shape[0])

for j in range(rkf.get_n_splits()):
        
    train_index, test_index = train_index_list[j], test_index_list[j]
    
    X_train= np.vstack(XX_task[train_index,:,:])
    Y_train = np.tile(labels, int(X_train.shape[0]/268.0))
    Y_train = label_binarize(Y_train,classes=[1,2,3,4,5,6,7,8,9])
    n_train = X_train.shape[0]
    
    X_test= np.vstack(XX_rest[test_index,:,:])
    Y_test = np.tile(labels, int(X_test.shape[0]/268.0))
    rois_ids= np.tile(np.arange(268), int(X_test.shape[0]/268.0))
    Y_test = label_binarize(Y_test,classes=[1,2,3,4,5,6,7,8,9])
    n_test = X_test.shape[0]
    
    X_train, Y_train=shuffle(X_train, Y_train, random_state=3000+j)
    
    X_test, Y_test,rois_ids=shuffle(X_test, Y_test, rois_ids, random_state=4000 + j)
    
    Y_true=np.argmax(Y_test,1)
    Yt=np.argmax(Y_train,1)
    
    #4. set model parameters
    num_classes = Y_test.shape[1]
    input_dim=X_train.shape[1]
    h_models = np.array(list_models)
    m=len(h_models)
    pz = patience
    
    for k in to_do :
        h_layer=h_models[k]

        test_nan=True
        time_start=time.clock()
        outfile = out_file + str(j)
        while test_nan :
            # 7. Define model architecture
            print('out_file=', outfile)
            model = Sequential()
            model.add(Dense(h_layer[0], input_dim=input_dim, activation='relu'))
            model.add(Dropout(drop))
            for i in range(1, len(h_layer)):
                model.add(Dense(h_layer[i], activation='relu'))
                model.add(Dropout(drop))
            model.add(Dense(num_classes, activation='softmax'))
            early_stopping = EarlyStopping(monitor='val_loss', patience=pz)
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss=loss, optimizer=sgd, metrics=[categorical_accuracy,'accuracy'])
            history=model.fit(X_train, Y_train, nb_epoch=epochs, batch_size=batch_size, callbacks=[early_stopping],validation_split=0.1,shuffle=True,verbose=2)
            score = model.evaluate(X_test, Y_test, batch_size=batch_size,verbose=0)
            test_nan=np.isnan(score[0])
            if test_nan:
                pz=4
                del model
        else:
            pz=patience
            scoret = model.evaluate(X_train, Y_train, batch_size=batch_size,verbose=0)
            print('test  %s = %5.2f%%' % (model.metrics_names[1], score[1] * 100))
            print('train %s = %5.2f%%' % (model.metrics_names[1], scoret[1] * 100))
            # Predict
            predicted=model.predict_classes(X_test,verbose=0)
            #predict probabilities
            predicted_p=model.predict(X_test,verbose=0) 
            predictedx=model.predict_classes(X_train,verbose=0)
            ss=0;
            sst=0;
            for i in range(n_test):
                if predicted[i]==Y_true[i]: ss=ss+1
            for i in range(n_train):
                if predictedx[i]==Yt[i]: sst=sst+1
            ss=ss/n_test
            sst=sst/n_train
            print(ss,sst)
            pd.DataFrame({'pred':predicted, 'true': Y_true}).to_csv(fold_model+outfile+"_pred.csv")
            np.save(fold_model+outfile + "_pred_probs", predicted_p)
            time_end=time.clock()
            cputime=time_end-time_start
            nepoch = len(history.history['acc'])
            print('cpu time = %6.2f sec nepoch= %d' % (cputime,nepoch))
            # serialize model to JSON
            model_json = model.to_json()
            with open(fold_model+outfile+".json", "w") as json_file: json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(fold_model+outfile+".h5")
            # save dataset in npz format
            print(X_test.shape)
            np.savez(fold_model+outfile+".npz",nepoch=nepoch,batch_size=batch_size,cputime=cputime,drop=drop,rois_ids=rois_ids)
            print("Saved data and model to disk on "+outfile)
        del model

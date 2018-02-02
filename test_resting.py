# 0. Select backend (theano or tensorflow) and outfile name for data and model
backend='theano'
out_file='task_tr_resting_ts_' 
batch_size = 128
epochs = 500
drop=0.2;
patience=20
loss='categorical_crossentropy'
seed = 125  # for reproducibility
fold_model = "./results/task_tr_rest_tr/"

#we saw previously this was the best model
list_models = [[500]]
to_do=range(len(list_models))

# 1. Import libraries and modules
import os
from scipy import io
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
from keras.models import Sequential
from keras import metrics
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle

labels = io.loadmat('./data/Shen268_yeo_RS7.mat',variable_names = 'yeoROIs',squeeze_me =True)['yeoROIs']

####################### TRAIN DATA#############################################
#load time series, calculate the correlation matrices, substract 1 from diagonal and and randomise input data
print('load time series, calculate the correlation matrices and substract 1 from diagonal for TASK DATA...')

array_dirs = sorted([f for f in os.listdir('./data/data_task_icafix') if f.startswith('sub-')])

corrs_list = [np.corrcoef(np.loadtxt('data/data_task_icafix/' + f + '/func_mean.txt')) for f in array_dirs]
corrs_list = [mat - np.identity(mat.shape[0]) for mat in corrs_list]

XX = np.vstack(corrs_list)

y = np.tile(labels, len(corrs_list))
TT = label_binarize(y,classes=[1,2,3,4,5,6,7,8,9])

n_train=int(XX.shape[0])

#shuffle data to disorder data
np.random.seed(seed)
index = np.arange(n_train)
index = shuffle(index)

X_train = XX[index,:].astype(np.float32)
Y_train = TT[index, :]

print( 'train data X shape: ', X_train.shape, ' and Y: ', Y_train.shape)


####################### TEST DATA##############################################
#load time series, calculate the correlation matrices, substract 1 from diagonal and and randomise input data
print('load time series, calculate the correlation matrices and substract 1 from diagonal for RESTING DATA...')

array_dirs = sorted([f for f in os.listdir('./data/data_fmri_clean') if f.startswith('sub-')])

corrs_list = [np.corrcoef(np.loadtxt('./data/data_fmri_clean/' + f + '/func_mean.txt')) for f in array_dirs]
corrs_list = [mat - np.identity(mat.shape[0]) for mat in corrs_list]

XX = np.vstack(corrs_list)

y = np.tile(labels, len(corrs_list))
TT = label_binarize(y,classes=[1,2,3,4,5,6,7,8,9])

n_test=int(XX.shape[0])

#same shuffling as before (examples belong to the same subject/node)
np.random.seed(seed)
index = np.arange(n_test)
index = shuffle(index)

np.save(fold_model + 'shuffl_ind_rest_test', index)

X_test = XX[index,:].astype(np.float32)
Y_test = TT[index, :]

print( 'test data X shape: ', X_test.shape, ' and Y: ', Y_test.shape)

Y_true=np.argmax(Y_test,1)
Yt=np.argmax(Y_train,1)
n_trials = range(1)

#4. set model parameters
num_classes = TT.shape[1]
input_dim=XX.shape[1]
h_models = np.array(list_models)
m=len(h_models)
pz = patience
for k in to_do :
    h_layer=h_models[k]
    for j in n_trials :
        test_nan=True
        time_start=time.clock()
        outfile = out_file+'_'+str(k)+str(j)
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
            model.compile(loss=loss, optimizer=sgd, metrics=[metrics.categorical_accuracy,'accuracy'])
            history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping],validation_split=0.1,shuffle=True,verbose=2)
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
            np.savez(fold_model+outfile+".npz",nepoch=nepoch,batch_size=batch_size,cputime=cputime,drop=drop,index=np.arange(XX.shape[0]))
            print("Saved data and model to disk on "+outfile)
        del model

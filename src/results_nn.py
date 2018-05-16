#Select outfile name for data and model and define constants of the neural networks
out_file='categorical'
batch_size = 128
epochs = 500
drop=0.2;
patience=20
loss='categorical_crossentropy'

# NN Models that we try
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


to_do= range(len(list_models)) 

fold_model = "./results/task_tr_task_test_nn_models/"
if not os.path.exists(fold_model):
    os.makedirs(fold_model)

input_dim=XX_task.shape[1]
h_models = np.array(list_models)
m=len(h_models)
pz = patience

for k in to_do:

    h_layer=h_models[k]

    for j in range(rkf.get_n_splits()):
        
        train_index, test_index = train_index_list[j], test_index_list[j]
        
        X_train= np.vstack(XX_task[train_index,:,:])
        Y_train = np.tile(labels, int(X_train.shape[0]/268.0))
        Y_train = label_binarize(Y_train,classes=[1,2,3,4,5,6,7,8,9])
        n_train = X_train.shape[0]
        
        X_test= np.vstack(XX_task[test_index,:,:])
        Y_test = np.tile(labels, int(X_test.shape[0]/268.0))
        Y_test = label_binarize(Y_test,classes=[1,2,3,4,5,6,7,8,9])
        n_test = X_test.shape[0]
        
        X_train, Y_train=shuffle(X_train, Y_train, random_state=1000+j)
        
        X_test, Y_test=shuffle(X_test, Y_test, random_state=2000 + j)
        
        Y_true=np.argmax(Y_test,1)
        Yt=np.argmax(Y_train,1)
	
        test_nan=True
        time_start=time.clock()
        outfile = out_file+'_'+str(k)+'_'+str(j)
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
            #pd.DataFrame(Y_true).to_csv("test_data.csv")
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
            #print(index)
            np.savez(fold_model+outfile+".npz",nepoch=nepoch,batch_size=batch_size,
                     cputime=cputime,drop=drop,index_train=train_index, index_test=test_index)
            print("Saved data and model to disk on "+outfile)
        
        del model

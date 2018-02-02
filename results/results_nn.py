# 0. Select backend (theano or tensorflow) and outfile name for data and model
backend='theano'
out_file='categorical'
batch_size = 128
epochs = 500
drop=0.2;
patience=20
loss='categorical_crossentropy'

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

num_classes = Y.shape[1]
input_dim=X.shape[1]
h_models = np.array(list_models)
m=len(h_models)
pz = patience

for k in to_do :

    h_layer=h_models[k]

    for j in n_shuffles:

        #np.random.shuffle(index)
        index = index_list[j]
        X = XX[index, :]
        Y = TT[index, :]
        
        	# 3. Get train and test end Preprocess input data
        n_train=int(n*trainpercentile)
        n_test=n-n_train
        X_train = X[0:n_train,:].astype(np.float32)
        X_test =  X[n_train:n,:].astype(np.float32)
        Y_train=Y[0:n_train,:]
        Y_test=Y[n_train:X.shape[0],:]
        Y_true=np.argmax(Y_test,1)
        Yt=np.argmax(Y_train,1)
	
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
            print(index)
            np.savez(fold_model+outfile+".npz",nepoch=nepoch,batch_size=batch_size,trainpercentile=trainpercentile,cputime=cputime,drop=drop,index=index)
            print("Saved data and model to disk on "+outfile)
        del model

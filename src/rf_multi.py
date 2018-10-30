fold_model = "./results/rf_multi_grid/"

if not os.path.exists(fold_model):
    os.makedirs(fold_model)
    
#Create the grid for hyperparameters
min_samp_splits = [2, 10, 50, 100]
max_features = [0.1, 0.25]
param_grid = model_selection.ParameterGrid({'min_samp': min_samp_splits, 
                                            'features': max_features})

np.save(fold_model + "param_grid.npy", param_grid)

accs = []
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
    
    
    df = pd.DataFrame(data={'true':Y_true})  
        
    for grid_id, hyp_param in enumerate(param_grid): 
        
        clf = ensemble.RandomForestClassifier(class_weight= 'balanced', 
                                      n_estimators= 500,
                                      min_samples_split = hyp_param['min_samp'],
                                      max_features= hyp_param['features'],
                                      n_jobs=-1, verbose = 1,
                                      random_state=0)

        clf.fit(X_train, Yt)
        
        yPred = clf.predict(X_test)
    
        
        pred_col = 'pred_grid_' + str(grid_id)  
        
        df = pd.concat([df, pd.DataFrame({pred_col: yPred})], axis=1) 
        
        print('Fold ', j, ' with max_depth ', str(hyp_param['min_samp']),
              'and max_fetures ', hyp_param['features'], ' completed ')
            
    print('Fold= ', j, ' completed ' )
        #j = j + 1
     
    df.to_csv(fold_model+'res_fold_' +str(j) + '.csv', index = False)

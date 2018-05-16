fold_model = "./results/rf_multi/"

if not os.path.exists(fold_model):
    os.makedirs(fold_model)
    
clf = ensemble.RandomForestClassifier(class_weight= 'balanced', n_estimators= 500, n_jobs=2, verbose = 1)

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
    
    feats = [np.zeros(268)]
    
    clf.fit(X_train, Yt)
    
    feats = np.append(feats,[clf.feature_importances_], axis=0)
    
    # save feature importances
    io.savemat(fold_model +'features_fold_' + str(j)+ '.mat', {'features': feats})
    
    yPred = clf.predict(X_test)
    acc = metrics.accuracy_score(Y_true, yPred)
    accs.append(acc) 
    
    df = pd.DataFrame(data={'true':Y_true ,'pred': yPred})
    
    #Save predictions
    df.to_csv(fold_model+'res_fold_' +str(j) + '.csv', index = False)
    print('Fold= ', j, ' completed ' )
    j = j + 1
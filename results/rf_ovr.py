#####################################################
#####################################################
fold_model = "./results/rf_ovr/"

if not os.path.exists(fold_model):  
    os.makedirs(fold_model)
    
clf = ensemble.RandomForestClassifier(class_weight= 'balanced', n_estimators= 500, n_jobs=1, verbose = 1)
mc= multiclass.OneVsRestClassifier(clf)

accs = []
for j in n_shuffle:

    index = index_list[j]
    X = XX[index, :]
    Y = TT[index, :]
    n_train=int(n*trainpercentile)
    n_test=n-n_train
    X_train = X[0:n_train,:].astype(np.float32)
    X_test =  X[n_train:n,:].astype(np.float32)
    Y_train=Y[0:n_train,:]
    Y_test=Y[n_train:X.shape[0],:]
    Y_true=np.argmax(Y_test,1)
    Yt=np.argmax(Y_train,1)
    
    feats = [np.zeros(268)]
    
    mc.fit(X_train, Yt)
    
    for estim in mc.estimators_:
        feats = np.append(feats,[estim.feature_importances_], axis=0)
    
    io.savemat(fold_model +'features_fold_' + str(j)+ '.mat', {'features': feats})
    
    yPred = mc.predict(X_test)
    acc = metrics.accuracy_score(Y_true, yPred)
    accs.append(acc) 
    
    df = pd.DataFrame(data={'true':Y_true ,'pred': yPred})
    
    df.to_csv(fold_model+'res_fold_' +str(j) + '.csv', index = False)
    #preds.append(mc.predict(Xtest))
    #trues.append(yTest)
    print('Fold= ', j, ' completed ' )
    j = j + 1



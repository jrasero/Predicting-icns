fold_model = "./results/svm_ovr_kernel/"

if not os.path.exists(fold_model):
    os.makedirs(fold_model)
    

alphas = 10.0**np.arange(-5,2) 
gammas  = 10.0**np.arange(-2,2)

param_grid = model_selection.ParameterGrid({'alpha': alphas,
                                            'gamma':gammas})

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
        
        nystroem = kernel_approximation.Nystroem(gamma=hyp_param['gamma'], 
                                                 random_state=0)
        
        clf =svm.LinearSVC(class_weight= 'balanced',
                     C=hyp_param['alpha'],
                     verbose = 1, 
                     random_state=10)

        clf.fit(nystroem.fit_transform(X_train),Yt)
    
        
        yPred = clf.predict(nystroem.transform(X_test))
        
        pred_col = 'pred_grid_' + str(grid_id)  
        df = pd.concat([df, pd.DataFrame({pred_col: yPred})], axis=1)  
#        print('Fold ', j, ' with alpha ', str(hyp_param['alpha']),
#              'and gamma ', str(hyp_param['gamma']), ' completed ')
#        
        
    df.to_csv(fold_model+'res_fold_' +str(j) + '.csv', index = False)
    print('Fold= ', j, ' completed ' )

print('SVM OVR kernel end...')

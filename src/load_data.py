 # for reproducibility when generating the folds
seed = 125 

#load time series, calculate the correlation matrices, substract 1 from diagonal and and randomize input data
print('load time series, calculate the correlation matrices and substract 1 from diagonal...')

array_dirs = sorted([f for f in os.listdir('./data/data_task_icafix') if f.startswith('sub-')])

corrs_list = [np.corrcoef(np.loadtxt('./data/data_task_icafix/' + f + '/func_mean.txt')) for f in array_dirs]
corrs_list = [mat - np.identity(mat.shape[0]) for mat in corrs_list]
XX_task = np.array(corrs_list)

labels = io.loadmat('./data/Shen268_yeo_RS7.mat',variable_names = 'yeoROIs',squeeze_me =True)['yeoROIs']

n_subjects= len(corrs_list)
num_classes = len(np.unique(labels))

#generate shuffles on subjects:
print(" ")
print('generating the shuffles to be used by all the models fitted..')

rkf=model_selection.RepeatedKFold(n_splits=10, n_repeats=5, random_state=seed)

index_subjects = range(n_subjects)
train_index_list=[]
test_index_list=[]
for train_index,test_index in rkf.split(index_subjects):
    train_index_list.append(train_index)
    test_index_list.append(test_index)
print(" ")

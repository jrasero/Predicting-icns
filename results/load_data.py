trainpercentile = 0.8
seed = 125  # for reproducibility
n_shuffles=range(5)

#load time series, calculate the correlation matrices, substract 1 from diagonal and and randomize input data
print('load time series, calculate the correlation matrices and substract 1 from diagonal...')

array_dirs = sorted([f for f in os.listdir('.data/data_task_icafix') if f.startswith('sub-')])

corrs_list = [np.corrcoef(np.loadtxt('.data/data_task_icafix/' + f + '/func_mean.txt')) for f in array_dirs]
corrs_list = [mat - np.identity(mat.shape[0]) for mat in corrs_list]

#Calculate observation matrix X and label vector...

XX = np.vstack(corrs_list)
labels = io.loadmat('./data/Shen268_yeo_RS7.mat',variable_names = 'yeoROIs',squeeze_me =True)['yeoROIs']

y = np.tile(labels, len(corrs_list))
TT = label_binarize(y,classes=[1,2,3,4,5,6,7,8,9])

n=int(XX.shape[0])
np.random.seed(seed)
index = np.arange(n)

#generate shuffles:
index_list=[]
for i in n_shuffles:
	index_list.append(shuffle(index, random_state=i)) 
    

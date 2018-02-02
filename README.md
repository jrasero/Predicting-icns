# Predicting functional networks from region connectivity profiles in task-based versus resting-state fMRI data

Scripts have to be run in the following way:

# download and preprocess resting fmri
1- sh shen_time_series_native_fmri_icafix.sh
# download and preprocess motor task fmri
2- sh shen_time_series_native_task_icafix.sh
# perform cross validation on task data
3- python cross_validation_main.py
# after best model picked, train on task and predict on resting
4- python test_resting.py
# generate the plots
5- python generate_plots.py

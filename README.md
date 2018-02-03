# Predicting functional networks from region connectivity profiles in task-based versus resting-state fMRI data


## Description:

Scripts have to be run in the following way:

1. Download and preprocess resting fmri:
 *sh shen_time_series_native_fmri_icafix.sh*
2. Download and preprocess motor task fmri:
 *sh shen_time_series_native_task_icafix.sh*
3. Perform cross validation on task data:
*python cross_validation_main.py*
4. Train on task and predict on resting after best model selected from previous step: 
*python test_resting.py*
5. Generate the plots
*python generate_plots.py*

Software required to run all the scripts:

python 2, numpy, pandas, scikit-learn, keras, nilearn

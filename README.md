# Predicting functional networks from region connectivity profiles in task-based versus resting-state fMRI data


## Description:
Code written in python and used for downloading and preprocessing the data and generating results and plots of the work:

**Predicting functional networks from region connectivity profiles in task-based versus resting-state fMRI data**  
*Javier Rasero, Hannelore Aerts, Marlis Ontivero Ortega, Jesus M. Cortes, Sebastiano Stramaglia, Daniele Marinazzo*  
bioRxiv 259077, that can be found in the link: https://doi.org/10.1371/journal.pone.0207385


## Software Requirements:

python 2.7, numpy, pandas, scikit-learn, keras 2.0, nilearn, fsl and FIX

## Usage:
Scripts have to be run in the following order (steps 1, 2 and 3 are to be run in a cluster given the time of computation):

1. Download and preprocess resting fmri:
 *sh shen_time_series_native_fmri_icafix.sh*
2. Download and preprocess motor task fmri:
 *sh shen_time_series_native_task_icafix.sh*
3. Perform cross validation on task data:
*python cross_validation_main.py*
4. Train on task and predict on resting after best model selected from previous step: 
*python test_resting.py*
5. Generate the plots (Figure 1 of the paper was included using Libreoffice Impress)
*python generate_main_plots.py* and *python generate_suppl_plots.py*


Please do not hesitate to contact us for any issue running the code, suggestions and remarks to jrasero.daparte@gmail.com



#!/bin/bash

subjectlist=s1200_unrelated_r227.txt

DBHOST=https://db.humanconnectome.org
PROJECT=HCP_1200
REST_URL_PREFIX=$DBHOST/data/archive/projects/$PROJECT/subjects

hcp_fix_dir=~/fix1.065

while read -r subject;
do

    echo "starting with subject $subject..."
    mkdir -p data/data_task_icafix/sub-$subject
    mkdir -p data/data_task_icafix/sub-$subject/temp

    temp_path=data/data_task_icafix/sub-$subject/temp
    subject_url_prefix=$REST_URL_PREFIX/$subject/experiments/${subject}_CREST/resources/${subject}_CREST/files

    file_relative_path=MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz
    output_file=data/data_task_icafix/sub-$subject/temp/tfMRI_MOTOR_LR.nii.gz

    if curl -u jrasero:mypassword -o $output_file --fail $subject_url_prefix/$file_relative_path; then
	echo "URL for file exist"
    else
	echo "$subject does not have file $file_relative_path" >> data_task_icafix/temp.log
	rm -r data/data_task_icafix/sub-$subject
	continue
    fi

    file_relative_path=MNINonLinear/xfms/standard2acpc_dc.nii.gz
    output_file=data/data_task_icafix/sub-$subject/temp/standard2acpc_dc.nii.gz

    if curl -u jrasero:mypassword -o $output_file --fail $subject_url_prefix/$file_relative_path; then
	echo "URL for file exist"
    else
	echo "$subject does not have file $file_relative_path" >> data_task_icafix/temp.log
	rm -r data/data_task_icafix/sub-$subject
	continue
    fi

    file_relative_path=T1w/T1w_acpc_dc.nii.gz
    output_file=data/data_task_icafix/sub-$subject/temp/T1w_acpc_dc.nii.gz

    if curl -u jrasero:mypassword -o $output_file --fail $subject_url_prefix/$file_relative_path; then
	echo "URL for file exist"
    else
	echo "$subject does not have file $file_relative_path" >> data_task_icafix/temp.log
	rm -r data/data_task_icafix/sub-$subject
	continue
    fi

    file_relative_path=MNINonLinear/Results/tfMRI_MOTOR_LR/Movement_Regressors.txt
    output_file=data/data_task_icafix/sub-$subject/temp/Movement_Regressors.txt

    if curl -u jrasero:mypassword -o $output_file --fail $subject_url_prefix/$file_relative_path; then
	echo "URL for file exist"
    else
	echo "$subject does not have file $file_relative_path" >> data/data_task_icafix/temp.log
	rm -r data/data_task_icafix/sub-$subject
	continue
    fi

    file_relative_path=MNINonLinear/T1w_restore_brain.nii.gz
    output_file=data/data_task_icafix/sub-$subject/temp/T1w_restore_brain.nii.gz


    if curl -u jrasero:mypassword -o $output_file --fail $subject_url_prefix/$file_relative_path; then
	echo "URL for file exist"
    else
	echo "$subject does not have file $file_relative_path" >> data/data_task_icafix/temp.log
	rm -r data_task_icafix/sub-$subject
	continue
    fi

    file_relative_path=MNINonLinear/wmparc.nii.gz
    output_file=data/data_task_icafix/sub-$subject/temp/wmparc.nii.gz

    if curl -u jrasero:mypassword -o $output_file --fail $subject_url_prefix/$file_relative_path; then
	echo "URL for file exist"
    else
	echo "$subject does not have file $file_relative_path" >> data/data_task_icafix/temp.log
	rm -r data/data_task_icafix/sub-$subject
	continue
    fi

    echo "applying FIX-ICA..."

    . $hcp_fix_dir/hcp_fix $temp_path/tfMRI_MOTOR_LR.nii.gz 2000

    echo "resampling T1 to 2mm resolution..."
    fsl5.0-flirt -in $temp_path/T1w_acpc_dc.nii.gz -ref $temp_path/T1w_acpc_dc.nii.gz -interp spline -applyisoxfm 2.0 -out $temp_path/T1w_acpc_dc_2mm.nii.gz

    echo "changing fMRI to this T1 volume space..."
    fsl5.0-applywarp -i $temp_path/tfMRI_MOTOR_LR_hp2000_clean.nii.gz -o $temp_path/tfMRI_MOTOR_LR_native.nii.gz -r $temp_path/T1w_acpc_dc_2mm.nii.gz \
-w $temp_path/standard2acpc_dc.nii.gz --interp=spline

    echo "trasforming shen partellation to native space..."
    fsl5.0-applywarp -i data/atlas/shen_2mm_268_parcellation.nii.gz -o $temp_path/shen_2mm_268_parcellation_native.nii.gz -r \
$temp_path/T1w_acpc_dc_2mm.nii.gz -w $temp_path/standard2acpc_dc.nii.gz --interp=nn

    echo "extracting roi time series..."
    fsl5.0-fslmeants -i $temp_path/tfMRI_MOTOR_LR_native.nii.gz -o data/data_task_icafix/sub-$subject/func_mean.txt \
--label=$temp_path/shen_2mm_268_parcellation_native.nii.gz --transpose

    echo "subject $subject finished. Deleting temporary files..."
    rm -r $temp_path

done < $subjectlist


#!/bin/sh

python preprocess.py [input_json_file] --dataset_name [name of the dataset] --summary_type [layman/expert]
python my_own_experiment.py --dataset_name [same as in previous line]
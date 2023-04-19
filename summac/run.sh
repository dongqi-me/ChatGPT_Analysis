#!/bin/sh

python calc_summacconv.py [model_output.json] [prefix of putput file] --summary_type [layman/expert]
# this if for evaluating the human summary; for model output change the 'sources' variable in calc_summacconv.py
import evaluate
import transformers
import os
import torch
from datasets import list_datasets, load_dataset
import nltk
import numpy as np
import pandas as pd

import pickle
import argparse
from summac.model_summac import SummaCConv
import json

def main(args):
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
    # print("model loaded")
    with open(args.model_output, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data = pd.read_csv(args.model_output)
    sources = [t['document'] for t in data]
    if args.summary_type == 'layman':
        preds = [t['layman_summary'] for t in data]
    elif args.summary_type == 'expert':
        preds = [t['expert_summary'] for t in data]

    else:
        raise ValueError("Please specify type of the summary: expert/layman")
    
    scores = [model_conv.score([source], [prediction])['scores'][0] for source, prediction in zip(sources, preds)]
    summacconv = {}
    summacconv['summacconv_list'] = scores
    summacconv['summacconv'] = round(np.mean(scores), args.round)
    print(np.mean(scores))
    pickle_file_name = args.prefix + '_summacconv.pkl'
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(summacconv, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='a json file containing model ouputs')
    parser.add_argument('prefix', type=str, default='result')
    # parser.add_argument('--dataset', type=str, help='elife/plos', default='elife')
    parser.add_argument('--summary_type', type=str, help='layman/expert', default='layman')
    parser.add_argument('--round', type=int, default=5)
    args = parser.parse_args()
    main(args)

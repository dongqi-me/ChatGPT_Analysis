from simalign import SentenceAligner
import nltk
import json
import pickle
import argparse
from tqdm import tqdm
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str) # a json file
    parser.add_argument('output_dir', type=str) # a pickle file
    args = parser.parse_args()

    with open(args.input_dir, 'r', encoding='utf-8') as f:
        outputs = json.load(f)

    data = {}
    length_pair = []
    mwmf_list = []
    inter_list = []
    itermax_list = []
    
    myaligner = SentenceAligner(model="bert", token_type="word", matching_methods="mai")

    for output in tqdm(outputs):
        src = output['input']
        trg = output['output']  # change the input here
        src_sentence = nltk.word_tokenize(src)
        trg_sentence = nltk.word_tokenize(trg)

        # The source and target sentences should be tokenized to words.
        src_sentence = ["This", "is", "a", "test", "."]
        trg_sentence = ["Das", "ist", "ein", "Test", "."]

        # The output is a dictionary with different matching methods.
        # Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
        alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)

        length_pair.append((len(src_sentence), len(trg_sentence)))
        mwmf_list.append(alignments['mwmf'])
        inter_list.append(alignments['inter'])
        itermax_list.append(alignments['itermax'])
        

    data['length_pair'] = length_pair
    data['mwmf'] = mwmf_list
    data['inter'] = inter_list
    data['itermax'] = itermax_list

    with open(args.output_dir, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
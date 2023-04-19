import textstat
import spacy
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from sentence_transformers import SentenceTransformer, util
from evaluate import load
sacrebleu = load("sacrebleu")
bertscore = load("bertscore")
rouge = load('rouge')
mauve = load('mauve')
meteor = load('meteor')
perplexity = load("perplexity", module_type="metric")
nlp = spacy.load("en_core_web_sm")
semantic_similarity_model = SentenceTransformer('all-mpnet-base-v2')

def text_readability(sequence):
    readability_results = {}
    readability_results["flesch_reading_ease"] = textstat.flesch_reading_ease(sequence)
    readability_results["coleman_liau_index"] = textstat.coleman_liau_index(sequence)
    readability_results["dale_chall_readability_score"] = textstat.dale_chall_readability_score(sequence)
    return readability_results

def text_formality(sequence):
    doc = nlp(sequence)
    noun = 0
    adjective = 0
    preposition = 0
    article = 0
    pronoun = 0
    verb = 0
    adverb = 0
    interjection = 0
    for token in doc:
        if token.pos_ == 'ADJ':
            adjective += 1
        elif token.pos_ == 'NOUN':
            noun += 1
        elif token.pos_ == 'VERB':
            verb += 1
        elif token.pos_ == 'ADV':
            adverb += 1
        elif token.pos_ == 'INTJ':
            interjection += 1
        elif token.pos_ == 'ADP':
            preposition += 1
        elif token.pos_ == 'DET':
            article += 1
    formality = (noun + adjective + preposition + article - pronoun - verb - adverb - interjection + 100) / 2
    return formality

def POS_distribution(seqence):
    doc = nlp(seqence)
    POS_dict = {}
    for token in doc:
      if token.pos_ in POS_dict.keys():
        POS_dict[token.pos_] +=1
      else:
        POS_dict[token.pos_] = 1
    return POS_dict

def DEP_distribution(seqence):
    doc = nlp(seqence)
    DEP_dict = {}
    for token in doc:
      if token.dep_ in DEP_dict.keys():
        DEP_dict[token.dep_] +=1
      else:
        DEP_dict[token.dep_] = 1
    return DEP_dict

def t_test(list1,list2):
    t_test = ttest_ind(list1, list1)
    return t_test

def ks_test(list1,list2):
    ks_test = ks_2samp(list1, list1)
    return ks_test

def semantic_similarity(sequence1, sequence2):
    embeddings1 = semantic_similarity_model.encode(sequence1)
    embeddings2 = semantic_similarity_model.encode(sequence2)
    similarity_score = util.cos_sim(embeddings1, embeddings2)
    return similarity_score.item()

def ngram_novelity(target: str, source: str, n: int):
    # First, split the target and source strings into lists of n-grams
    target_words = target.split(' ')
    source_words = source.split(' ')
    target_ngrams = [target_words[i:i+n] for i in range(len(target_words) - n + 1)]
    source_ngrams = [source_words[i:i+n] for i in range(len(source_words) - n + 1)]
    num_novel_ngrams = sum(1 for ngram in target_ngrams if ngram not in source_ngrams)

    if len(target_ngrams) == 0:
        return 0.0
    return num_novel_ngrams / len(target_ngrams)

def do_rouge(predictions, references):
    rouge_results = rouge.compute(predictions=predictions, references=references)
    return rouge_results

def do_meteor(predictions, references):
    meteor_results = meteor.compute(predictions=predictions, references=references)
    return meteor_results

def do_perplexity(predictions, references):
    perplexity_results = perplexity.compute(predictions=predictions, references=references)
    return perplexity_results


# %% [markdown]
# ### Imports and libraries

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# %%
from sacrebleu import sentence_bleu, corpus_bleu
from comet import download_model, load_from_checkpoint
from evaluate import load

# %%
import ntpath
import random
import json
import os
import string
import time
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# %%
# utils
from utils.commonutils import load_samples, make_dir, get_random_name, append_config_to_file, lang_abbr_to_lang_code, lang_abbr_to_lang
from utils.utils_data import get_train_test_data
from utils.constants import *
from model_parameters import model_parameters

# prompt construction
from prompts import get_n_shots, construct_zero_shot, construct_prompt

# Preprocessing prompts, batching prompts and post processing outputs
from MTDataset import MTDataset
from process_outputs import predict_outputs
from preprocess_prompts import handle_repetitive_examples

# scoring functions
from scoring_functions import init_comet_computation, init_comet_qe_20_computation, init_comet_da_22_computation, init_chrf
from scoring_functions import get_chrf_scores, get_comet_scores, get_comet_mean_score, get_comet_qe_20_scores, get_comet_da_22_scores

# helper functions
from helper_functions import read_recommendations, get_samples, clear_gpu_memory, get_model

# %% [markdown]
# ### Constants

# %%
BATCH_SIZE = 8

RANKINGS_LABSE = 'rankings_labse'
RANKINGS_BM25 = 'rankings_bm25'
RANKINGS_BM25_AND_RERANKING = 'rankings_bm25_and_reranking'
RANKINGS_LABSE_AND_RERANKING = 'rankings_labse_and_reranking'
RANKINGS_KNN_ROBERTA = 'rankings_knn_roberta'
RANKINGS_COMET_QA = 'rankings_comet_qa'
RANKINGS_BM25_AND_LABSE = 'rankings_bm25_and_labse'
RANKINGS_BM25_AND_CHRF = 'rankings_bm25_and_chrf'
RANKINGS_BM25_AND_3_WAY = 'rankings_bm25_and_3_way'
RANKINGS_BM25_AND_3_WAY_ORACLE = 'rankings_bm25_and_3_way_oracle'
RANKINGS_BM25_REGRESSION = 'rankings_bm25_regression'
RANKINGS_BM25_AND_DIVERSITY = 'rankings_bm25_and_diversity'
RANKINGS_REGRESSION = 'rankings_regression'
RANDOM_SELECTION = 'random_selection'

# Constants related to Dataset
SAMANANTAR = 'samanantar'
FLORES = 'flores'

# %% [markdown]
# ### Initiating Scoring functions

# %%
chrf = init_chrf()
comet_da_20_metric = init_comet_computation()
comet_qe_20_metric = init_comet_qe_20_computation()
comet_da_22_metric = init_comet_da_22_computation()

# %% [markdown]
# ### Function to generate MT and evaluating translation

# %%
# This function evaluates the BLOOM model and also captures the MT outputs
def get_bleu_scores(pipe, mp: model_parameters, experiment=''):
    model_name = mp.name.split('/')[1]
    
    # languages for which the model should be evaluated
    src_lang = lang_abbr_to_lang.get(mp.src_lang) 
    dst_lang = lang_abbr_to_lang.get(mp.dst_lang)

    # create output directory
    output_dir, prompts_dir = 'outputs', 'prompts'
    make_dir(output_dir)
    make_dir(prompts_dir)

    # make note of configuration
    scores_file = '{}/scores.csv'.format(output_dir)
    msg = '{} [{}]\n'.format(str(mp).strip(), experiment)
    append_config_to_file(scores_file, msg=msg)
    print(mp)

    # get train/test samples
    src_train_samples, dst_train_samples, src_test_samples, dst_test_samples = get_samples(mp.training_source, mp.testing_source, mp.src_lang, mp.dst_lang)

    # get ranking of dev samples if reranking flag is true
    if mp.has_reranking:
        rankings = read_recommendations(mp.strategy, mp.training_source, mp.testing_source, mp.src_lang, mp.dst_lang, mp.strategy_nested)
        if len(rankings) == 0:
            print('No ranking found for: {}'.format(src_lang))
            return

    # capture configuration and generate random name for file to map the configuration
    random_name = get_random_name()
    prediction_file = '{}/{}_{}_{}_{}_{}_shots_pred_{}.txt'.format(output_dir, experiment, model_name, src_lang, dst_lang, mp.no_of_shots, random_name)

    # create an object to batch the examples
    datasetObj = MTDataset()
    
    # all prompts
    prompts = ''

    for qid, input_sample in enumerate(src_test_samples):
        
        recommendations = []
        if mp.has_reranking:
            recommendations = rankings[str(qid)]
            
            # # COMET_QE_QUERY_DST_SCORE, COMET_QE_SRC_DST_SCORE
            # if mp.strategy_nested == COMET_QE_QUERY_DST_SCORE \
            # or mp.strategy_nested == COMET_QE_SRC_DST_SCORE:
            #     recommendations.sort(key=lambda x: x[mp.strategy_nested], reverse=True)
            #     recommendations = list(map(lambda x: x["index"], recommendations))
            #     # print(recommendations)
            # elif mp.strategy_nested == SRC_DST_PPL\
            # or mp.strategy_nested == SRC_DST_QUERY_PPL:
            #     recommendations.sort(key=lambda x: x[mp.strategy_nested])
            #     recommendations = list(map(lambda x: x["index"], recommendations))
            #     # print(recommendations)
            # elif mp.strategy_nested == NO_OF_TOKENS_IN_SRC_SENT \
            # or mp.strategy_nested == NO_OF_TOKENS_IN_DST_SENT:
            #     recommendations.sort(key=lambda x: x[mp.strategy_nested], reverse=True)
            #     recommendations = list(map(lambda x: x["index"], recommendations))

            # recommendation structure has been changed
            if mp.strategy == RANKINGS_BM25 \
            or mp.strategy == RANKINGS_LABSE \
            or mp.strategy == RANKINGS_COMET_QA \
            or mp.strategy == RANKINGS_BM25_AND_LABSE \
            or mp.strategy == RANKINGS_BM25_AND_LABSE_QUERY_DST \
            or mp.strategy == RANKINGS_BM25_AND_LABSE_SRC_DST \
            or mp.strategy == RANKINGS_BM25_AND_CHRF \
            or mp.strategy == RANKINGS_BM25_AND_3_WAY \
            or mp.strategy == RANKINGS_BM25_AND_3_WAY_ORACLE \
            or mp.strategy == RANKINGS_REGRESSION \
            or mp.strategy == RANKINGS_3WAY_REGRESSION \
            or mp.strategy == RANKINGS_NO_OF_TOKENS \
            or mp.strategy == RANKINGS_CUSTOM \
            or mp.strategy == RANKINGS_LINEAR_REGRESSION:
                # recommendations are in [{ "index": 630729, "score": 37.21}, ... ]
                recommendations = list(map(lambda x: x["index"], recommendations))

            # tries to take different prompt examples
            if mp.diversify_prompts:
                recommendations = recommendations[0::10]

            # Remove the repetitive prompts. Also smaller sentences are present ahead of bigger sentences 
            # in BM25 ranking. 
            # if mp.training_source == SAMANANTAR:
            recommendations = handle_repetitive_examples(src_train_samples, dst_train_samples, recommendations)
            
            # take recommendations as many as the no of shots
            recommendations = recommendations[:mp.no_of_shots]
            # changes the order of prompts (low-score to high-score examples)
            if mp.inc_reranking:
                recommendations.reverse()

        # prompt construction
        if mp.no_of_shots > 1:
            shots = get_n_shots(mp, src_train_samples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=recommendations)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang)
        elif mp.no_of_shots == 0:
            content = construct_zero_shot(input_sample, src_lang, dst_lang)
        elif mp.no_of_shots == 1:
            shots = get_n_shots(mp, src_train_samples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=recommendations)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=1)
        
        prompts = prompts + '{}\n{}\n\n\n'.format(qid, content)

        datasetObj.addprompt(content)
        datasetObj.addinput(input_sample)

    # write prompts to file
    with open('{}/{}_{}_{}.txt'.format(prompts_dir, experiment, mp.src_lang, mp.dst_lang), 'w') as f:
        f.write(prompts)
        
    # obtained the output from model
    pred_dst = predict_outputs(pipe, datasetObj, prediction_file, mp.name) 
    # print(pred_dst)

    # obtain the bleu score
    blue_score = corpus_bleu(pred_dst, [dst_test_samples]).score
    blue_score = round(blue_score, 2)
    print('BLEU score -> {}'.format(blue_score))

    # obtain comet score
    comet_score = get_comet_mean_score(predicted=pred_dst, references=dst_test_samples, source=src_test_samples, comet_da_20_metric=comet_da_20_metric)
    print('COMET score -> {}'.format(comet_score))

    comet_qe_20_scores = get_comet_qe_20_scores(predicted=pred_dst, source=src_test_samples, comet_qe_20_metric=comet_qe_20_metric)
    comet_qe_20_scores = list(map(lambda x: round(x, 4), comet_qe_20_scores))
    comet_qe_20_score = round(np.mean(comet_qe_20_scores), 4)
    print('comet_qe_20_score score -> {}'.format(comet_qe_20_score))


    comet_da_22_scores = get_comet_da_22_scores(predicted=pred_dst, references=dst_test_samples, source=src_test_samples, comet_da_22_metric=comet_da_22_score)
    comet_da_22_scores = list(map(lambda x: round(x, 4), comet_da_22_scores))
    comet_da_22_score = round(np.mean(comet_da_22_scores), 4)
    print('comet_da_22_score score -> {}'.format(comet_da_22_score))

    
    # obtain chrf and chrf++ score
    chrf_score, chrfpp_score = get_chrf_scores(pred_dst, dst_test_samples, chrf)
    print('chrF score -> {}, chrF++ score -> {}'.format(chrf_score, chrfpp_score))

    with open(scores_file, 'a') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(model_name, mp.type_of_algo, src_lang, dst_lang, mp.no_of_shots, blue_score, comet_score, chrf_score, chrfpp_score, comet_qe_20_score, comet_da_22_score, mp.use_8_bit, random_name))

# %% [markdown]
# ### Evaluation

# %%
# name = "facebook/opt-6.7b", BLOOM_3B, BLOOM_7B, XGLM_7B
name = BLOOM_7B

# parameters for the model
mp = model_parameters(name=name)

# must use 8-bit inferencing if it is XGLM
# also make sure we use transformers==4.28.1
if name == XGLM_7B:
    mp.use_8_bit=True
    
# generate pipe and use the same pipe instead of creating one each time
pipe = get_model(mp.name, type_of_algo=mp.type_of_algo, use_8_bit=mp.use_8_bit)

# %%
mp.training_source=SAMANANTAR
mp.testing_source=FLORES
mp.has_reranking=True
mp.inc_reranking=True
mp.no_of_shots=4

experiment = 'exp_120_test'

mp.strategy = RANKINGS_BM25
# mp.strategy_nested = COMET_QE_20_REGRESSION

mp.src_lang=BEN_BENG
mp.dst_lang=ENG_LATN
get_bleu_scores(pipe, mp, experiment='{}.1'.format(experiment))

# mp.src_lang=GUJ_GUJR
# mp.dst_lang=ENG_LATN
# get_bleu_scores(pipe, mp, experiment='{}.2'.format(experiment))

# mp.src_lang=HIN_DEVA
# mp.dst_lang=ENG_LATN
# get_bleu_scores(pipe, mp, experiment='{}.3'.format(experiment))

# mp.src_lang=ENG_LATN
# mp.dst_lang=BEN_BENG
# get_bleu_scores(pipe, mp, experiment='{}.4'.format(experiment))

# mp.src_lang=ENG_LATN
# mp.dst_lang=GUJ_GUJR
# get_bleu_scores(pipe, mp, experiment='{}.5'.format(experiment))

# mp.src_lang=ENG_LATN
# mp.dst_lang=HIN_DEVA
# get_bleu_scores(pipe, mp, experiment='{}.6'.format(experiment))



# %%
# # get bleu score
# mp.training_source=EUROPARL
# mp.testing_source=FLORES
# mp.has_reranking=True
# mp.inc_reranking=True
# mp.no_of_shots=4

# mp.strategy = RANKINGS_NO_OF_TOKENS
# mp.strategy_nested = NO_OF_TOKENS_IN_SRC_SENT

# mp.src_lang=FRA_LATN
# mp.dst_lang=ENG_LATN
# get_bleu_scores(pipe, mp, experiment='{}.7'.format(experiment))

# mp.src_lang=DEU_LATN
# mp.dst_lang=ENG_LATN
# get_bleu_scores(pipe, mp, experiment='{}.8'.format(experiment))

# mp.src_lang=ENG_LATN
# mp.dst_lang=FRA_LATN
# get_bleu_scores(pipe, mp, experiment='{}.9'.format(experiment))

# mp.src_lang=ENG_LATN
# mp.dst_lang=DEU_LATN
# get_bleu_scores(pipe, mp, experiment='{}.10'.format(experiment))


# %%
# # get bleu score
# mp.training_source=PARACRAWL
# mp.testing_source=FLORES
# mp.has_reranking=True
# mp.inc_reranking=True
# mp.no_of_shots=4

# mp.src_lang=RUS_CYRL
# mp.dst_lang=ENG_LATN
# get_bleu_scores(pipe, mp, experiment='{}.11'.format(experiment))

# mp.src_lang=ENG_LATN
# mp.dst_lang=RUS_CYRL
# get_bleu_scores(pipe, mp, experiment='{}.12'.format(experiment))




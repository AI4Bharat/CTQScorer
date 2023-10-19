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
# ### Initiating Scoring functions

# %%
chrf = init_chrf()
comet_da_20_metric = init_comet_computation()
comet_qe_20_metric = init_comet_qe_20_computation()
comet_da_22_metric = init_comet_da_22_computation()

# %% [markdown]
# ### Functions to generate MT and evaluating translation

# %%
def update_nested_strategy_based_on_strategy(mp: model_parameters):
    if mp.strategy in [COMET_QE_QUERY_SRC_SCORE, COMET_QE_QUERY_DST_SCORE, COMET_QE_SRC_DST_SCORE]:
        mp.strategy_nested = mp.strategy
        mp.strategy = RANKINGS_COMET_QA
    elif mp.strategy in [LABSE_SCORE_QUERY_SRC, LABSE_SCORE_QUERY_DST, LABSE_SCORE_SRC_DST]:
        mp.strategy_nested = mp.strategy
        mp.strategy = RANKINGS_BM25_AND_3_WAY
    elif mp.strategy in [SRC_DST_PPL, SRC_DST_QUERY_PPL]:
        mp.strategy_nested = mp.strategy
        mp.strategy = RANKINGS_BM25_AND_PERPLEXITY
    elif mp.strategy in [NO_OF_TOKENS_IN_SRC_SENT, NO_OF_TOKENS_IN_DST_SENT]:
        mp.strategy_nested = mp.strategy
        mp.strategy = RANKINGS_NO_OF_TOKENS
    return mp

# %%

# This function generates MT and evaluates the translation using the model 
# specified in model parameters and also applies ranking strategy.
def get_bleu_scores(pipe, mp: model_parameters, experiment=''):
    model_name = mp.name.split('/')[1]
    
    # languages for which the model should be evaluated
    src_lang = lang_abbr_to_lang.get(mp.src_lang) 
    dst_lang = lang_abbr_to_lang.get(mp.dst_lang)

    # create output directory
    output_dir, prompts_dir = 'outputs', 'prompts'
    make_dir(output_dir)
    make_dir(prompts_dir)

    # updating nested strategy helps to read from right recommendation file
    mp = update_nested_strategy_based_on_strategy(mp)

    # make note of configuration
    scores_file = '{}/scores.csv'.format(output_dir)
    msg = '{} [{}]\n'.format(str(mp).strip(), experiment)
    append_config_to_file(scores_file, msg=msg)
    print(mp)

    # get train/test samples
    src_train_samples, dst_train_samples, src_test_samples, dst_test_samples = get_samples(mp.training_source, mp.testing_source, mp.src_lang, mp.dst_lang)

    # get ranking of dev samples if reranking flag is true
    if mp.has_reranking:
        rankings = read_recommendations(mp.strategy, mp.training_source, mp.testing_source, mp.src_lang, mp.dst_lang)
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
            
            # inside nested strategies, there can be multiple different ways to choose reranking            
            if mp.strategy_nested:
                if mp.strategy_nested in [COMET_QE_QUERY_SRC_SCORE, COMET_QE_QUERY_DST_SCORE, COMET_QE_SRC_DST_SCORE]:
                    recommendations.sort(key=lambda x: x[mp.strategy_nested], reverse=True)
                elif mp.strategy_nested in [LABSE_SCORE_QUERY_SRC, LABSE_SCORE_QUERY_DST, LABSE_SCORE_SRC_DST]:
                    recommendations.sort(key=lambda x: x[mp.strategy_nested], reverse=True)
                elif mp.strategy_nested in [SRC_DST_PPL, SRC_DST_QUERY_PPL]:
                    recommendations.sort(key=lambda x: x[mp.strategy_nested])
                elif mp.strategy_nested in [NO_OF_TOKENS_IN_SRC_SENT, NO_OF_TOKENS_IN_DST_SENT]:
                    recommendations.sort(key=lambda x: x[mp.strategy_nested], reverse=True)
                recommendations = list(map(lambda x: x["index"], recommendations))
            elif mp.strategy == RANKINGS_BM25_AND_RERANKING:
                pass 
            elif mp.strategy in [RANKINGS_BM25, RANKINGS_BM25_AND_CHRF, RANKINGS_BM25_AND_3_WAY, 
                                 RANKINGS_3WAY_REGRESSION, RANKINGS_REGRESSION, RANKINGS_LINEAR_REGRESSION]:
                # recommendations are in [{ "index": 630729, "score": 37.21}, ... ]
                recommendations = list(map(lambda x: x["index"], recommendations))
            else:
                print('Invalid strategy: {}'.format(mp.strategy))
                return

            # tries to take different prompt examples
            if mp.diversify_prompts:
                recommendations = recommendations[0::10]

            # Remove the repetitive examples
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
# ### Machine Translation and Evaluation

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
mp.strategy = RANKINGS_BM25
mp.src_lang=BEN_BENG
mp.dst_lang=ENG_LATN

experiment = 'exp_120_test'
get_bleu_scores(pipe, mp, experiment='{}.1'.format(experiment))



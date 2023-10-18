# %% [markdown]
# ### Imports and libraries

# %%
# %pip install bitsandbytes
# %pip install git+https://github.com/huggingface/transformers.git
# %pip install accelerate

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import XGLMTokenizer, XGLMForCausalLM
import torch
from transformers import pipeline

# %%
import numpy as np
from evaluate import load
import matplotlib.pyplot as plt

# %%
# from nltk.translate.bleu_score import sentence_bleu
from sacrebleu import sentence_bleu, corpus_bleu

# %%
import ntpath
import random
import json
import os
import string
import time
import logging

# %%
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# %%
# imports for script unification tasks
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from indicnlp.transliterate.unicode_transliterate import ItransTransliterator
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import euclidean_distances
from utils import get_embeddings, get_cos_sim_for_embeddings
from utils_language import romanize_sentences, script_convert_sentences
from cross_lingual import cross_lingual_recommendations

# %%
# utils
from utils import load_samples, make_dir, get_random_name, append_config_to_file
from utils_language import lang_abbr_to_lang_code, lang_abbr_to_lang
from utils_language import configure_indic_nlp_library
configure_indic_nlp_library()

# %%
from model_parameters import model_parameters
from constants import *

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
IN22_OTHER_SOURCES = 'in22_other_sources'
IN22_CONVERSATIONS = 'in22_conversations'
IN22_WIKIPEDIA = 'in22_wikipedia'

# %% [markdown]
# ### Helper functions

# %%
# reads the recommendations obtained from BM25 and reranking algorithm
def read_recommendations(strategy, training_source, testing_source, src_lang, dst_lang, strategy_nested=''):
    json_data = {}

    if strategy_nested == '':
        recommendations = 'recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang)
        ranking_file_name = '{}/{}'.format(strategy, recommendations)
        with open(ranking_file_name, 'r') as f:
            json_data = json.load(f)
    else:
        recommendations = 'recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang)
        ranking_file_name = '{}/{}/{}'.format(strategy, strategy_nested, recommendations)
        with open(ranking_file_name, 'r') as f:
            json_data = json.load(f)
    
    return json_data

# %%

chrf = load("chrf")
def get_chrf_scores(predicted, references):
    tmp_references = []
    for reference in references:
        tmp_references.append([reference])
    references = tmp_references

    chrfscore = chrf.compute(predictions=predicted, references=references).get('score')
    chrfpp_score = chrf.compute(predictions=predicted, references=references, word_order=2).get('score')
    chrfscore = round(chrfscore, 2)
    chrfpp_score = round(chrfpp_score, 2)
    return chrfscore, chrfpp_score

# %%
# Fix to get around torch error for computing comet score
from comet import download_model, load_from_checkpoint
def init_comet_computation():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # comet_metric = load('comet' , 'Unbabel/wmt20-comet-da')
    
    model_path = download_model("Unbabel/wmt20-comet-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric

comet_da_20_metric = init_comet_computation()

def get_comet_scores(predicted, references, source):
    comet_metric = comet_da_20_metric
    scores = []

    # sometimes we just run for 5 to 10 samples
    k = len(predicted)
    references = references[:k]
    source = source[:k]

    idx = 0
    while idx < len(predicted):
        batch = int(min(1024, len(predicted) - idx))
        predicted_batch = predicted[idx: idx + batch]
        references_batch = references[idx: idx + batch]
        source_batch = source[idx: idx + batch]

        data = []
        for src, mt, ref in zip(source_batch, predicted_batch, references_batch):
            data.append({
                "src": src,
                "mt": mt,
                "ref": ref
            })
        
        comet_score = comet_metric.predict(data, progress_bar=True)        
        # comet_score = comet_metric.compute(predictions=predicted_batch, references=references_batch, sources=source_batch, progress_bar=True)
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores

# %%
def get_comet_mean_score(predicted, references, source):
    scores = get_comet_scores(predicted, references, source)
    mean_score = np.mean(scores)
    # print(len(scores))
    mean_score = round(mean_score, 4)
    return mean_score

# %%
def init_comet_qe_20_computation():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # comet_metric = load('comet' , 'Unbabel/wmt20-comet-da')
    
    model_path = download_model("Unbabel/wmt20-comet-qe-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric

comet_qe_20_metric = init_comet_qe_20_computation()
def get_comet_qe_20_scores(predicted, source):
    comet_metric = comet_qe_20_metric
    scores = []

    # sometimes we just run for 5 to 10 samples
    k = len(predicted)
    source = source[:k]

    idx = 0
    while idx < len(predicted):
        batch = int(min(1024, len(predicted) - idx))
        predicted_batch = predicted[idx: idx + batch]
        source_batch = source[idx: idx + batch]

        data = []
        for src, mt in zip(source_batch, predicted_batch):
            data.append({
                "src": src,
                "mt": mt,
            })
        
        comet_score = comet_metric.predict(data, progress_bar=True)        
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores

# %%
def init_comet_da_22_computation():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # comet_metric = load('comet' , 'Unbabel/wmt20-comet-da')
    
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric

comet_da_22_metric = init_comet_da_22_computation()
def get_comet_da_22_scores(predicted, references, source):
    comet_metric = comet_da_22_metric
    scores = []

    # sometimes we just run for 5 to 10 samples
    k = len(predicted)
    references = references[:k]
    source = source[:k]

    idx = 0
    while idx < len(predicted):
        batch = int(min(1024, len(predicted) - idx))
        predicted_batch = predicted[idx: idx + batch]
        references_batch = references[idx: idx + batch]
        source_batch = source[idx: idx + batch]

        data = []
        for src, mt, ref in zip(source_batch, predicted_batch, references_batch):
            data.append({
                "src": src,
                "mt": mt,
                "ref": ref
            })
        
        comet_score = comet_metric.predict(data, progress_bar=True)        
        # comet_score = comet_metric.compute(predictions=predicted_batch, references=references_batch, sources=source_batch, progress_bar=True)
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores

# %% [markdown]
# ### Different types of prompt construction

# %%
def get_n_mixed_shots(mp: model_parameters, src_samples, dst_samples, script_converted_samples, n_shots, src_lang, dst_lang, itr_lang, recommendations=[], to_english=True):
    random.seed(mp.seed)
    random_numbers = recommendations
    THRESHOLD = 120
    
    # If no recommendations are present, then generate random numbers
    while(len(random_numbers) < n_shots):
        x = random.randint(0,len(src_samples) - 1)
        sent = src_samples[x].strip('"').split()
        if x in random_numbers or len(sent) > THRESHOLD:
            continue
        random_numbers.append(x)

    content_org_sample = ''
    content_converted_sample = ''
    count = 0
    i = 0
    while count < n_shots and i < len(random_numbers):
        sent = src_samples[random_numbers[i]].strip('"').split()
        src_sample = src_samples[random_numbers[i]].strip('"')
        dst_sample = dst_samples[random_numbers[i]].strip('"')
        script_converted_sample = script_converted_samples[random_numbers[i]].strip('"')

        # TODO: Figure out if [Hindi Sentence], [Gujarati sentence] for script conversions make a difference
        if len(sent) < THRESHOLD:
            if to_english:            
                # get samples from hin sentences
                if count % 2 == 0:
                    content_org_sample = content_org_sample + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, src_sample, dst_lang, dst_sample)
                
                # get samples from hindi sentences which are script converted to gujarati
                else:
                    content_converted_sample = content_converted_sample + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, script_converted_sample, dst_lang, dst_sample)

            else:
                # get samples from eng-hin sentences
                if count % 2 == 0:
                    content_org_sample = content_org_sample + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, src_sample, dst_lang, dst_sample)
                
                # get samples from hindi sentences which are script converted to gujarati
                else:
                    content_converted_sample = content_converted_sample + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, src_sample, dst_lang, script_converted_sample)
            
            count += 1

        i += 1

    return content_org_sample + content_converted_sample

# %%
# returns n-shot example for the given source and target languages.
# we can pass recommendations for an input sample obtained from BM25 & reranking algorithm.
def get_n_shots(mp: model_parameters, src_samples, dst_samples, n_shots, src_lang, dst_lang, recommendations=[], transliterate_flags=(False,'','')):

    # start_time = time.time()
    # sometimes the recommendations from BM25 is less than n-shots
    # then we randomly choose samples from the dev dataset
    random.seed(mp.seed)
    random_numbers = recommendations

    # Don't add sentences larger than 120 words
    THRESHOLD = 120
    for random_number in random_numbers:
        sent = src_samples[random_number].strip('"').split()
        if len(sent) > THRESHOLD:
            random_numbers.remove(random_number)

    while(len(random_numbers) < n_shots):
        x = random.randint(0,len(src_samples) - 1)
        sent = src_samples[x].strip('"').split()
        if x in random_numbers or len(sent) > THRESHOLD:
            continue
        random_numbers.append(x)

    content = ''

    count = 0
    i = 0
    while count < n_shots and i < len(random_numbers):
        sent = src_samples[random_numbers[i]].strip('"').split()
        src_sample = src_samples[random_numbers[i]].strip('"')
        dst_sample = dst_samples[random_numbers[i]].strip('"')

        if transliterate_flags[0]:
            src_sample = UnicodeIndicTransliterator.transliterate(src_sample, transliterate_flags[1], transliterate_flags[2])

        if len(sent) < THRESHOLD:
            count += 1
            if n_shots == 1:
                content = content + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, src_sample, dst_lang, dst_sample)
            else:
                content = content + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, src_sample, dst_lang, dst_sample)
        i += 1

    return content

# %%
def get_shot_from_input(ind, src_test_samples, dst_test_samples, n_shots, src_lang, dst_lang):
    content = """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, src_test_samples[ind], dst_lang, dst_test_samples[ind])
    return content

# %%


# %%
# This function concatenates the n-shots and the given input sample
def construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=0):
    if n_shots == 1:
        return shots + """{} Sentence: "{}"
{} Sentence: """.format(src_lang, input_sample.strip('"'), dst_lang)
    return shots + """{} Sentence: "{}"
{} Sentence: """.format(src_lang, input_sample.strip('"'), dst_lang)

# %%
# This function generates zero shot example
def construct_zero_shot(input_sample, src_lang, dst_lang):
    return """Translate {} Sentence: "{}" to {} Sentence: """.format(src_lang, input_sample.strip('"'), dst_lang)

# %%
# This function returns the model based on the arguments we pass.
# If use_8_bit is true the function returns the quantizied 8-bit model.
def get_model(model_name, type_of_algo='Greedy', use_8_bit=False):
    
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    model = "facebook/opt-30b"

    model_kwargs = {"device_map": "auto", "load_in_8bit": True}
    m = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    generator = pipeline(task="text-generation", model=m, tokenizer=tokenizer)
    """
    # Use 8-bit
    model_kwargs = {"device_map": "auto"}
    if use_8_bit:
        model_kwargs= {"device_map": "auto", "load_in_8bit": True}

    pipe = None

    if model_name == XGLM_7B:
        model = AutoModelForCausalLM.from_pretrained(XGLM_7B, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(XGLM_7B, use_fast=False)
        tokenizer.padding_side = "left"
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer,
                        return_full_text=False, early_stopping=True)
    else:
        # torch_dtype=torch.float32
        if type_of_algo == 'Greedy' or type_of_algo == '':
            pipe = pipeline(model=model_name, model_kwargs=model_kwargs, 
            return_full_text=False, early_stopping=True)
        elif type_of_algo == 'Beam':
            pipe = pipeline(model=model_name, model_kwargs=model_kwargs,
            num_beams=3, return_full_text=False, early_stopping=True)
        
    return pipe

# %%
def get_shots_from_cross_lingual(recommendations, inc_order=False):
    content = ''
    if inc_order:
        recommendations.reverse()
    for recommendation in recommendations:
        src_sample = recommendation[0]
        dst_sample = recommendation[1]
        src_lang_code = recommendation[2]
        dst_lang_code = 'eng_Latn'

        content = content + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(lang_abbr_to_lang.get(src_lang_code), src_sample, lang_abbr_to_lang.get(dst_lang_code), dst_sample)
    return content

# %%
from utils_data import get_train_test_data
def get_samples(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset=False, intr_lang=''):
    train_src_path, train_dst_path, test_src_path, test_dst_path = get_train_test_data(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset)
    src_train_samples = load_samples(train_src_path)
    dst_train_samples = load_samples(train_dst_path)
    src_test_samples = load_samples(test_src_path)
    dst_test_samples = load_samples(test_dst_path)

    # consider intermediate language in case
    if intr_lang != '':
        if training_source == 'flores':
            intr_train_path = 'dataset/train/{}.dev'.format(intr_lang)
        elif training_source == 'samanantar':
            intr_lang = lang_abbr_to_lang_code.get(intr_lang)
            intr_train_path = 'dataset/samanantar/en-{}/train.{}'.format(intr_lang, intr_lang)
        intr_lang_samples = load_samples(intr_train_path)
        return src_train_samples, dst_train_samples, src_test_samples, dst_test_samples, intr_lang_samples

    return src_train_samples, dst_train_samples, src_test_samples, dst_test_samples

# %% [markdown]
# ### Experiment BM25, LaBSE and other example selection algorithms

# %%
"""
Class to run the input samples in a batch
"""
class MyDataset(Dataset):
    
    def __init__(self):
        self.prompts = []
        self.inputs = []
    
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def addprompt(self, prompt):
        self.prompts.append(prompt)

    def addinput(self, input):
        self.inputs.append(input)

    def getNumTokens(self, start_index):
        inputs_in_batch = self.inputs[start_index : start_index + BATCH_SIZE]
        inputs_in_batch = list(map(lambda x: x.split(), inputs_in_batch))
        tokens_in_input_batch = list(map(lambda x: len(x), inputs_in_batch))
        max_tokens = max(tokens_in_input_batch)

        return int(max_tokens * 1.5)

# %%
def post_process_output(output):
    output = output.split('\n###')[0]
    output = output.split('###')[0]
    output = output.strip().strip('"').replace('\n', '')
    return output

"""
Given the model and an object with prompts, this function
predicts the outputs in a batch and writes it to the file
and returns the list of outputs.
"""
def predict_outputs(pipe, datasetObj, prediction_file, model_name='', experiment_flags=('','','')):
    start_time = time.time()

    pred_dst = []
    isOPTModel = 'opt' in model_name

    for start_index in tqdm(range(0, len(datasetObj), BATCH_SIZE)):
        
        for output in pipe(datasetObj.prompts[start_index: start_index + BATCH_SIZE], 
        max_new_tokens=datasetObj.getNumTokens(start_index), batch_size=BATCH_SIZE):
            
            # clean up the output obtained from model for evaluation purpose
            output = post_process_output(output[0].get('generated_text'))
            
            # for en-xx script converting only when examples have non-dst script
            if experiment_flags[0] == 'exp_48' or experiment_flags[0] == 'exp_49':
                output = script_convert_sentences([output], experiment_flags[1], experiment_flags[2])[0]

            # print(output)
            pred_dst.append(output)

            # capture output obtained from model
            with open(prediction_file, 'a') as f:
                f.write('{}\n'.format(output))
        
    end_time = time.time()
    print('Time for one set: {}'.format(round(end_time - start_time, 3)))
    return pred_dst

# %%
from collections import Counter
import re

def has_similar_no_of_tokens(existing, current):
    matched = {key: min(existing[key], current[key]) for key in current if key in existing }
    total_no_tokens_matched = sum(matched.values())
    total_no_tokens_in_current = sum(current.values())
    # For some reason we get empty strings some times
    if total_no_tokens_in_current == 0:
        return True
    similarity = total_no_tokens_matched / total_no_tokens_in_current
    # print(similarity)
    return True if similarity >= 0.8 else False

def check_similar_sent_exists_in_group(existing_group, current):
    for existing in existing_group:
        if has_similar_no_of_tokens(existing, current) or has_similar_no_of_tokens(current, existing):
            return True
    return False

def handle_repetitive_examples(src_train_samples, dst_train_samples, recommendations):
    filtered_indexes = []
    src_group = []
    dst_group = []
    
    for index in recommendations:
        src_sent = re.sub('[?,.!।]+', '', src_train_samples[index].lower())
        dst_sent = re.sub('[?,.!।]+', '', dst_train_samples[index].lower())

        src_tokens_counter = dict(Counter(src_sent.split()))
        dst_tokens_counter = dict(Counter(dst_sent.split()))
        
        if check_similar_sent_exists_in_group(src_group, src_tokens_counter) \
        or check_similar_sent_exists_in_group(dst_group, dst_tokens_counter):
            continue
        else:
            src_group.append(src_tokens_counter)
            dst_group.append(dst_tokens_counter)
            filtered_indexes.append(index)
    
    return filtered_indexes

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
    # '{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(name, type_of_algo, max_new_tokens, use_8_bit, toEnglish, num_of_shots, reranking, dataset, rec_source)
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

    # capture configuration and generate random name for file to map the configuration
    random_name = get_random_name()
    prediction_file = '{}/{}_{}_{}_{}_{}_shots_pred_{}.txt'.format(output_dir, experiment, model_name, src_lang, dst_lang, mp.no_of_shots, random_name)

    # create an object to batch the examples
    datasetObj = MyDataset()
    
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
    comet_score = get_comet_mean_score(predicted=pred_dst, references=dst_test_samples, source=src_test_samples)
    print('COMET score -> {}'.format(comet_score))

    comet_qe_20_scores = get_comet_qe_20_scores(predicted=pred_dst, source=src_test_samples)
    comet_qe_20_scores = list(map(lambda x: round(x, 4), comet_qe_20_scores))
    comet_qe_20_score = round(np.mean(comet_qe_20_scores), 4)
    print('comet_qe_20_score score -> {}'.format(comet_qe_20_score))


    comet_da_22_scores = get_comet_da_22_scores(predicted=pred_dst, references=dst_test_samples, source=src_test_samples)
    comet_da_22_scores = list(map(lambda x: round(x, 4), comet_da_22_scores))
    comet_da_22_score = round(np.mean(comet_da_22_scores), 4)
    print('comet_da_22_score score -> {}'.format(comet_da_22_score))

    
    # obtain chrf and chrf++ score
    chrf_score, chrfpp_score = get_chrf_scores(pred_dst, dst_test_samples)
    print('chrF score -> {}, chrF++ score -> {}'.format(chrf_score, chrfpp_score))

    with open(scores_file, 'a') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(model_name, mp.type_of_algo, src_lang, dst_lang, mp.no_of_shots, blue_score, comet_score, chrf_score, chrfpp_score, comet_qe_20_score, comet_da_22_score, mp.use_8_bit, random_name))

# %% [markdown]
# ### Script Unification
# 
# Consider an LLM which has been pretrained with languages in various
# scripts including Bengali (bn), Meitei (mni) and English (en).
# Consider the task is to translate from mni to en.
# But we have only test data for mni to en and no parallel training data for mni-en.
# So we retrieve from bn to en dataset.
# We can use LaBSE as our retriever.
# We script-convert mni to bn and then search in bn-en training dataset. The
#  assumption here is that mni and bn are close and the retrieved results
# are similar to the test input.
# We now have two options: (1) we can use the bn-en retrieved examples as
# prompt; (2) we can use the script converted bn to mni paired with en as
# the prompt.
# We compare between these two settings.

# %%
def retrive_similar_sentences(query_sentences, dev_sentences, k=32):
    # generate embeddings
    query_embeddings = get_embeddings(query_sentences)
    dev_embeddings = get_embeddings(dev_sentences)

    # calculate the nearest euclidean distance
    euclidean_dists = euclidean_distances(query_embeddings, dev_embeddings)
    rows, cols = euclidean_dists.shape

    # sort scores and return the closest samples
    scores = {}
    only_indexes = {}
    for i in range(rows):
        for j in range(cols):
            if i in scores:
                scores[i].append({"doc_id": j, "score": round(float(euclidean_dists[i][j]), 3)})
            else:
                scores[i] = [{"doc_id": j, "score": round(float(euclidean_dists[i][j]), 3)}]

        scores[i] = sorted(scores[i], key=lambda d: d['score']) 
        only_indexes[i] = [doc["doc_id"] for doc in scores[i][:k]]
    
    return scores, only_indexes

# %%
def get_bloom_scores_using_script_unification(pipe, mp: model_parameters, src_lang_code='', dst_lang_code='', intr_lang_code='', experiment='', script_convert_for_retrival=False):
    
    # model name
    model_name = mp.name.split('/')[1]

    # get language using language code
    src_lang = lang_abbr_to_lang.get(src_lang_code)
    intr_lang = lang_abbr_to_lang.get(intr_lang_code)
    dst_lang = lang_abbr_to_lang.get(dst_lang_code)

    src_train_samples, dst_train_samples, src_test_samples, dst_test_samples, intr_train_examples = get_samples(mp.training_source, mp.testing_source, 
                                                                src_lang_code, dst_lang_code, is_ranking_for_devset=False, intr_lang=intr_lang_code)

    # script convert sentences from src_lang to intr_lang
    script_converted_query_sents = script_convert_sentences(src_test_samples, src_lang_code, intr_lang_code)

    rankings = []
    if mp.has_reranking:
        if script_convert_for_retrival:
            scores, rankings = retrive_similar_sentences(script_converted_query_sents, intr_train_examples)
        else:
            # print('using labse example ranking')
            scores, rankings = retrive_similar_sentences(src_test_samples, intr_train_examples)

    # create output directory
    output_dir = 'outputs'
    make_dir(output_dir)

    # make note of configuration
    scores_file = '{}/scores.csv'.format(output_dir)
    configuration = '[Script Unification] {}, {}, {}, {}, {}\n'.format(src_lang_code, intr_lang_code, dst_lang_code, str(mp).strip(), experiment)
    append_config_to_file(scores_file, configuration)

    # capture configuration
    random_name = get_random_name()
    prediction_file = '{}/{}_{}_{}_{}_{}_{}_shots_{}_pred_{}.txt'.format(output_dir, experiment, model_name, mp.type_of_algo, src_lang_code, dst_lang_code, mp.no_of_shots, mp.use_8_bit, random_name)
    
    # create an object to batch the examples
    datasetObj = MyDataset()

    # in the case of hin-eng, hin releated examples are converted to guj
    script_converted_intr_train_examples = script_convert_sentences(intr_train_examples, intr_lang_code, src_lang_code)
    romanized_intr_train_examples = romanize_sentences(intr_train_examples, intr_lang_code)

    # in the case of other3: guj side of guj-eng is script converted to deva
    script_converted_train_examples = script_convert_sentences(src_train_samples, src_lang_code, intr_lang_code)

    # in the case of other4: guj side of guj-eng is romanized
    romanized_train_examples = romanize_sentences(src_train_samples, src_lang_code)

    # in the case of exp_38: kan side of eng-kan is script converted to deva
    script_convert_dst_train_examples_to_intr = script_convert_sentences(dst_train_samples, dst_lang_code, intr_lang_code)

    # in the case of exp_40: hin side of eng-hin is script converted to kan
    script_convert_intr_train_examples_to_dst = script_convert_sentences(intr_train_examples, intr_lang_code, dst_lang_code)

    # in the case of exp_42 or exp_43 compute the best samples from cross lingual
    if experiment == 'exp_42' or experiment == 'exp_43':
        cross_lingual_ecommendations = cross_lingual_recommendations(src_lang_code)

    for qid, input_sample in enumerate(src_test_samples):
        recommendations = []
        if mp.has_reranking:
            recommendations = rankings[qid]

        # change in direction en-xx
        if experiment == 'exp_48':
            shots = get_n_shots(mp, src_train_samples, script_convert_dst_train_examples_to_intr, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_49':
            shots = get_n_shots(mp, src_train_samples, intr_train_examples, mp.no_of_shots, src_lang, intr_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_50':
            shots = get_n_shots(mp, src_train_samples, script_convert_intr_train_examples_to_dst, mp.no_of_shots, src_lang, intr_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_51':
            shots = get_n_mixed_shots(mp, src_train_samples, intr_train_examples, script_convert_intr_train_examples_to_dst, mp.no_of_shots, src_lang, intr_lang, dst_lang, recommendations=[], to_english=False)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)

        elif experiment == 'other3':
            shots = get_n_shots(mp, script_converted_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            input_sample = script_convert_sentences([input_sample], src_lang_code, intr_lang_code)[0]
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_3.1':
            shots = get_n_shots(mp, script_converted_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_3.2' or experiment == 'exp_27':
            shots = get_n_shots(mp, romanized_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'other4' or experiment == 'exp_28':
            shots = get_n_shots(mp, romanized_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            input_sample = romanize_sentences([input_sample], src_lang_code)[0]
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'other5':
            shots = get_n_shots(mp, intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'other6':
            shots = get_n_shots(mp, intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            input_sample = script_convert_sentences([input_sample], src_lang_code, intr_lang_code)[0]
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'other7':
            shots = get_n_shots(mp, script_converted_intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_7.1':
            shots = get_n_mixed_shots(mp, intr_train_examples, dst_train_samples, script_converted_intr_train_examples, mp.no_of_shots, intr_lang, dst_lang, src_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'other8':
            shots = get_n_shots(mp, romanized_intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'other9':
            shots = get_n_shots(mp, romanized_intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            input_sample = romanize_sentences([input_sample], src_lang_code)[0]
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_9.1':
            shots = get_n_shots(mp, script_converted_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_9.2':
            shots = get_n_shots(mp, script_converted_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        
        # experiments for Panjabi
        elif experiment == 'exp_9.3':
            shots = get_n_shots(mp, src_train_samples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_9.4':
            shots = get_n_shots(mp, script_converted_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_9.5':
            shots = get_n_shots(mp, intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_9.6':
            shots = get_n_shots(mp, script_converted_intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)

        elif experiment == 'other22':
            shots = get_n_shots(mp, intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=recommendations)
            input_sample = script_convert_sentences([input_sample], src_lang_code, intr_lang_code)[0]
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'other23':
            shots = get_n_shots(mp, script_converted_intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=recommendations)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)

        # experiments for Kannada
        elif experiment == 'exp_32':
            shots = get_n_shots(mp, script_converted_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_33':
            shots = get_n_shots(mp, intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_34':
            shots = get_n_shots(mp, script_converted_intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)      
        elif experiment == 'exp_38':
            shots = get_n_shots(mp, src_train_samples, script_convert_dst_train_examples_to_intr, mp.no_of_shots, src_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_39':
            shots = get_n_shots(mp, src_train_samples, intr_train_examples, mp.no_of_shots, src_lang, intr_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_40':
            shots = get_n_shots(mp, src_train_samples, script_convert_intr_train_examples_to_dst, mp.no_of_shots, src_lang, intr_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)

        # misc experiment
        elif experiment == 'exp_41':
            shots = get_shot_from_input(qid, src_test_samples, dst_test_samples, mp.no_of_shots, src_lang, dst_lang)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_42':
            shots = get_shots_from_cross_lingual(cross_lingual_ecommendations[qid], inc_order=False)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_43':
            shots = get_shots_from_cross_lingual(cross_lingual_ecommendations[qid], inc_order=True)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)

        # french, spanish experiments
        elif experiment == 'exp_46' or experiment == 'exp_46.1' or experiment == 'exp_46.2' or experiment == 'exp_46.3':
            shots = get_n_shots(mp, intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_47':
            # https://github.com/valentinmace/noisy-text for noisy text
            noisy_eng_examples = load_samples('dataset/noisy/eng_Latn.noisy')
            shots = get_n_shots(mp, noisy_eng_examples, dst_train_samples, mp.no_of_shots, 'Some', dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp_47.2':
            noisy_hin_examples = load_samples('dataset/noisy/hin_Deva.noisy')
            shots = get_n_shots(mp, noisy_hin_examples, dst_train_samples, mp.no_of_shots, 'Some', dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
            
        elif experiment == 'exp5':
            # exp5: Use random hin-eng examples as prompts
            shots = get_n_shots(mp, intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=[])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp4':
            # exp4: using pseudo english-english sentences
            shots = get_n_shots(mp, dst_train_samples, dst_train_samples, mp.no_of_shots, "Some", dst_lang, recommendations=recommendations)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp3':
            # exp3: retrive hin-eng similar examples and script convert hin side to guj script and choose 2 from each
            shots = get_n_mixed_shots(mp, intr_train_examples, dst_train_samples, script_converted_intr_train_examples, mp.no_of_shots, intr_lang, dst_lang, src_lang, recommendations=recommendations)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp2':
            # exp2: retrive hin-eng similar examples and script convert hin side to guj script
            shots = get_n_shots(mp, script_converted_intr_train_examples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=recommendations)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)
        elif experiment == 'exp1':
            # exp2: retrive hin-eng similar examples and use them finally with a guj script
            shots = get_n_shots(mp, intr_train_examples, dst_train_samples, mp.no_of_shots, intr_lang, dst_lang, recommendations=recommendations)
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=mp.no_of_shots)

        # print(content)
        datasetObj.addprompt(content)
        datasetObj.addinput(input_sample)

    # obtained the output from model
    pred_dst = predict_outputs(pipe, datasetObj, prediction_file, mp.name, experiment_flags=(experiment, intr_lang_code, dst_lang_code))

    # print(pred_dst) 

    # obtain the bleu score and comet score
    blue_score = corpus_bleu(pred_dst, [dst_test_samples]).score
    blue_score = round(blue_score, 2)
    comet_score = get_comet_mean_score(predicted=pred_dst, references=dst_test_samples, source=src_test_samples)
    print('COMET score -> {}'.format(comet_score))
    

    print('Model -> {}, Type -> {}, Number of shots -> {}, BLEU score -> {}, COMET score -> {}, Use 8 bit -> {}'.format(model_name, mp.type_of_algo, mp.no_of_shots, blue_score, comet_score, mp.use_8_bit))
    with open(scores_file, 'a') as f:
        f.write('{},{},{},{},{},{},{},{},{}\n'.format(model_name, mp.type_of_algo, src_lang, dst_lang, mp.no_of_shots, blue_score, mp.use_8_bit, comet_score, random_name))

# %% [markdown]
# ### Model initialization and inferencing

# %%
def clear_gpu_memory():
    # clear cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# # Regression Idea

# %%
# This function evaluates the BLOOM model and also captures the MT outputs
def get_prompt_scores(pipe, mp: model_parameters, experiment=''):
    model_name = mp.name.split('/')[1]
    
    # languages for which the model should be evaluated
    src_lang = lang_abbr_to_lang.get(mp.src_lang)
    dst_lang = lang_abbr_to_lang.get(mp.dst_lang)

    # create output directory
    output_dir, prompts_dir = 'outputs', 'prompts'
    make_dir(output_dir)
    make_dir(prompts_dir)

    # make note of configuration
    # '{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(name, type_of_algo, max_new_tokens, use_8_bit, toEnglish, num_of_shots, reranking, dataset, rec_source)
    scores_file = '{}/scores.csv'.format(output_dir)
    msg = '{} [{}]\n'.format(str(mp).strip(), experiment)
    append_config_to_file(scores_file, msg=msg)
    print(mp)

    # load samples from samanantar corpus
    src_train_samples, dst_train_samples, src_flores_dev_samples, dst_flores_dev_samples = get_samples(mp.training_source, mp.testing_source,
                                                                                        mp.src_lang, mp.dst_lang, is_ranking_for_devset=True)

    # get ranking of dev samples if reranking flag is true
    if mp.has_reranking:
        rankings = read_recommendations(mp.strategy, mp.training_source, mp.testing_source, mp.src_lang, mp.dst_lang)
        if len(rankings) == 0:
            print('No ranking found for: {}'.format(src_lang))

    # capture configuration and generate random name for file to map the configuration
    random_name = get_random_name()
    prediction_file = '{}/{}_{}_{}_{}_{}_shots_pred_{}.txt'.format(output_dir, experiment, model_name, src_lang, dst_lang, mp.no_of_shots, random_name)

    # write prompts to file
    with open('{}/{}_{}_{}.txt'.format(prompts_dir, experiment, mp.src_lang, mp.dst_lang), 'w') as f:
        f.write('')

    for qid, input_sample in enumerate(tqdm(src_flores_dev_samples)):

        # all prompts
        prompts = ''
            
        recommendations = []
        if mp.has_reranking:
            recommendations = rankings[str(qid)]

            # recommendation structure has been changed
            if mp.strategy == RANKINGS_BM25_REGRESSION:
                # recommendations are in [{ "index": 630729, "score": 37.21}, ... ]
                recommendations = list(map(lambda x: x["index"], recommendations))

        # create an object to batch the examples
        datasetObj = MyDataset()

        for recommendation in recommendations:

            # prompt construction
            shots = get_n_shots(mp, src_train_samples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[recommendation])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=1)
            
            prompts = prompts + '{}\n{}\n\n\n'.format(qid, content)

            # print(content)
            # print('\n\n\n')
            datasetObj.addprompt(content)
            datasetObj.addinput(input_sample)
    
        # write prompts to file
        with open('{}/{}_{}_{}.txt'.format(prompts_dir, experiment, mp.src_lang, mp.dst_lang), 'a') as f:
            f.write(prompts)

        # obtained the output from model
        pred_dst = predict_outputs(pipe, datasetObj, prediction_file, mp.name) 
        # print(pred_dst)

        # obtain comet score
        refs = [dst_flores_dev_samples[qid]] * len(pred_dst)
        srcs = [src_flores_dev_samples[qid]] * len(pred_dst)
        # print(refs)
        # print(srcs)
        comet_scores = get_comet_scores(predicted=pred_dst, references=refs, source=srcs)
        comet_scores = list(map(lambda x: round(x, 4), comet_scores))
        # print('COMET score -> {}'.format(comet_scores))

        bleu_scores = []
        ref = dst_flores_dev_samples[qid]
        for candidate in pred_dst:
            bleu_scores.append(sentence_bleu(candidate, [ref]).score)
        bleu_scores = list(map(lambda x: round(x, 2), bleu_scores))
        # print('BLEU scores -> {}'.format(bleu_scores))

        # write scores to regression file
        regression_scores_file = '{}/regression_scores_{}_{}.csv'.format(output_dir, mp.src_lang, mp.dst_lang)
        for elem_id_in_corpus, comet_score, bleu_score in zip(recommendations, comet_scores, bleu_scores):
            with open(regression_scores_file, 'a') as f:
                f.write('{},{},{},{}\n'.format(qid, elem_id_in_corpus, comet_score, bleu_score))
                

# %% [markdown]
# ### Evaluation

# %%
# name = "facebook/opt-6.7b", BLOOM_3B, BLOOM_7B, XGLM_7B
name = XGLM_7B

# parameters for the model
mp = model_parameters(name=name)

# must use 8-bit inferencing if it is XGLM
# also make sure we use transformers==4.28.1
if name == XGLM_7B:
    mp.use_8_bit=True
    
# generate pipe and use the same pipe instead of creating one each time
pipe = get_model(mp.name, type_of_algo=mp.type_of_algo, use_8_bit=mp.use_8_bit)

# %%
# mp.training_source=SAMANANTAR
# mp.testing_source=FLORES
# mp.has_reranking=True
# mp.inc_reranking=True
# mp.no_of_shots=4

experiment = 'exp_120'

mp.strategy = RANKINGS_CUSTOM
mp.strategy_nested = COMET_QE_20_REGRESSION

# mp.src_lang=BEN_BENG
# mp.dst_lang=ENG_LATN
# get_bleu_scores(pipe, mp, experiment='{}.1'.format(experiment))

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
# get bleu score
mp.training_source=EUROPARL
mp.testing_source=FLORES
mp.has_reranking=True
mp.inc_reranking=True
mp.no_of_shots=4


mp.src_lang=FRA_LATN
mp.dst_lang=ENG_LATN
get_bleu_scores(pipe, mp, experiment='{}.7'.format(experiment))

mp.src_lang=DEU_LATN
mp.dst_lang=ENG_LATN
get_bleu_scores(pipe, mp, experiment='{}.8'.format(experiment))

mp.src_lang=ENG_LATN
mp.dst_lang=FRA_LATN
get_bleu_scores(pipe, mp, experiment='{}.9'.format(experiment))

mp.src_lang=ENG_LATN
mp.dst_lang=DEU_LATN
get_bleu_scores(pipe, mp, experiment='{}.10'.format(experiment))


# %%
# get bleu score
mp.training_source=PARACRAWL
mp.testing_source=FLORES
mp.has_reranking=True
mp.inc_reranking=True
mp.no_of_shots=4

mp.src_lang=RUS_CYRL
mp.dst_lang=ENG_LATN
get_bleu_scores(pipe, mp, experiment='{}.11'.format(experiment))

mp.src_lang=ENG_LATN
mp.dst_lang=RUS_CYRL
get_bleu_scores(pipe, mp, experiment='{}.12'.format(experiment))




from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import torch

from utils.utils_data import get_train_test_data
from utils.commonutils import load_samples
from utils.constants import BLOOM_7B, XGLM_7B, EXAMPLE_SELECTION_TEST_DATA, EXAMPLE_SELECTION_TRAIN_DATA


# reads the recommendations obtained from BM25 and reranking algorithm
def read_recommendations(strategy, training_source, testing_source, src_lang, dst_lang, is_train_data=False):
    json_data = {}

    if is_train_data:
        recommendations = 'recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang)
        ranking_file_name = '{}/{}/{}'.format(EXAMPLE_SELECTION_TRAIN_DATA, strategy, recommendations)
        with open(ranking_file_name, 'r') as f:
            json_data = json.load(f)
    else:
        recommendations = 'recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang)
        ranking_file_name = '{}/{}/{}'.format(EXAMPLE_SELECTION_TEST_DATA, strategy, recommendations)
        with open(ranking_file_name, 'r') as f:
            json_data = json.load(f)
    
    return json_data


def get_samples(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset=False):
    train_src_path, train_dst_path, test_src_path, test_dst_path = get_train_test_data(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset)
    src_train_samples = load_samples(train_src_path)
    dst_train_samples = load_samples(train_dst_path)
    src_test_samples = load_samples(test_src_path)
    dst_test_samples = load_samples(test_dst_path)
    
    return src_train_samples, dst_train_samples, src_test_samples, dst_test_samples


def clear_gpu_memory():
    # clear cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    
# This function returns the model based on the arguments we pass.
def get_model(model_name, type_of_algo='Greedy', use_8_bit=False):

    model_kwargs = {"device_map": "auto"}
    if use_8_bit:
        model_kwargs= {"device_map": "auto", "load_in_8bit": True}

    pipe = None
    if model_name == XGLM_7B:
        model = AutoModelForCausalLM.from_pretrained(XGLM_7B, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(XGLM_7B, use_fast=False)
        tokenizer.padding_side = "left"
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, return_full_text=False, early_stopping=True)
    else:
        if type_of_algo == 'Greedy' or type_of_algo == '':
            pipe = pipeline(model=model_name, model_kwargs=model_kwargs, return_full_text=False, early_stopping=True)
        elif type_of_algo == 'Beam':
            pipe = pipeline(model=model_name, model_kwargs=model_kwargs, num_beams=3, return_full_text=False, early_stopping=True)
        
    return pipe
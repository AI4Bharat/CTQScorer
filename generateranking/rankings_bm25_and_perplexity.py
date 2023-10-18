from utils import load_samples, write_json_to_file
from utils.constants import *
from library import Perplexity

import logging
import json
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer

def isNaN(num):
    return num!= num

def get_ppl_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset=False, use_xglm_model=False):
    queries = load_samples(test_src_path)
    logging.info('loading corpus...')
    training_src_examples = load_samples(train_src_path)
    training_dst_examples = load_samples(train_dst_path)
    logging.info('loaded corpus...')
    logging.info('number of samples is: {}'.format(len(training_src_examples)))

    json_data = ''
    with open(bm25_file_name, 'r') as f:
        json_data = json.load(f)
    
    MODEL_NAME = XGLM_7B if use_xglm_model else BLOOM_7B
    logging.info('Model is: {}'.format(MODEL_NAME))
    
    # load perplexity model
    if MODEL_NAME == BLOOM_7B:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", load_in_8bit=False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    elif MODEL_NAME == XGLM_7B:
        # we gotta load the model in 8-bit precision and set padding left
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        tokenizer.padding_side = "left"        
        
    perplexity = Perplexity()
    batch_size=4
    
    result = {}
    for qid, query in enumerate(queries):
        logging.info('qid: {}'.format(qid))
        src_dst_data = []
        src_dst_query_data = []
        src_dst_ppls = []
        src_dst_query_ppls = []
        
        bm25_rankings = json_data[str(qid)]
        bm25_rankings = list(map(lambda x: x["index"], bm25_rankings))
        for index in bm25_rankings:
            src_dst = '{} {}'.format(training_src_examples[index], training_dst_examples[index])
            src_dst_query = '{} {}'.format(src_dst, query)            
            if len(training_src_examples[index]) == 0:
                training_src_examples[index] = 'aaaaaaa'
            if len(training_dst_examples[index]) == 0:
                training_dst_examples[index] = 'aaaaaaa'
            src_dst_data.append(src_dst)
            src_dst_query_data.append(src_dst_query)        
            
        max_length = max(list(map(lambda x: len(x), src_dst_data)))
        model.config.max_length = max_length
        src_dst_ppls = perplexity._compute(data=src_dst_data, model=model, tokenizer=tokenizer, batch_size=batch_size)["perplexities"]
        
        max_length = max(list(map(lambda x: len(x), src_dst_query_data)))
        model.config.max_length = max_length
        src_dst_query_ppls = perplexity._compute(data=src_dst_query_data, model=model, tokenizer=tokenizer, batch_size=batch_size)["perplexities"]
        
        ranking = []
        for (index, src_dst_ppl, src_dst_query_ppl) in zip(bm25_rankings, src_dst_ppls, src_dst_query_ppls):
            src_dst_ppl = src_dst_ppl if not isNaN(src_dst_ppl) else 999999
            src_dst_query_ppl = src_dst_query_ppl if not isNaN(src_dst_query_ppl) else 999999
            
            ranking.append({"index": index,
                            "src_dst_ppl": round(float(src_dst_ppl), 2),
                            "src_dst_query_ppl": round(float(src_dst_query_ppl), 2)
                            })

        if not is_ranking_for_devset:
            # Lower the ppl, better the result
            ranking.sort(key=lambda x: x['src_dst_ppl'])
        result[qid] = ranking

        write_json_to_file(result, 'tmp.json')

    return result

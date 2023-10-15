from utils import load_samples, init_logging, make_dir, lang_abbr_to_lang_code, get_train_test_data
import logging
import json


def get_no_of_tokens(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset=False):
    queries = load_samples(test_src_path)
    logging.info('loading corpus...')
    training_src_samples = load_samples(train_src_path)
    training_dst_samples = load_samples(train_dst_path)
    logging.info('loaded corpus...')
    logging.info('number of samples is: {}'.format(len(training_src_samples)))

    json_data = ''
    with open(bm25_file_name, 'r') as f:
        json_data = json.load(f)
        
    result = {}
    for qid, query in enumerate(queries):
        logging.info('qid: {}'.format(qid))
        data = []
        bm25_rankings = json_data[str(qid)]
        bm25_rankings = list(map(lambda x: x["index"], bm25_rankings))

        no_of_tokens_in_query = len(query.split())
        no_of_tokens_in_src_sents = []
        no_of_tokens_in_dst_sents = []
        for index in bm25_rankings:
            no_of_tokens_in_src_sents.append(len(training_src_samples[index].split()))
            no_of_tokens_in_dst_sents.append(len(training_dst_samples[index].split()))

        # arrange indexes based on the no of tokens
        ranking = []
        for (index, no_of_tokens_in_src_sent, no_of_tokens_in_dst_sent) in zip(bm25_rankings, no_of_tokens_in_src_sents, no_of_tokens_in_dst_sents):
            ranking.append({"index": index, 
                            "no_of_tokens_in_query": no_of_tokens_in_query, 
                            "no_of_tokens_in_src_sent": no_of_tokens_in_src_sent,
                            "no_of_tokens_in_dst_sent": no_of_tokens_in_dst_sent,
                            })
        
        result[qid] = ranking

    return result
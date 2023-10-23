from utils import load_samples
import logging
from retriv import SearchEngine
import random


def get_collection_data_structure(samples):
    collection = []
    for id, sample in enumerate(samples):
        current = {"id": id, "text": sample }
        collection.append(current)
    return collection


def get_bm25_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang):    
    queries = load_samples(test_src_path)
    logging.info('loading corpus...')
    training_samples = load_samples(train_src_path)
    logging.info('loaded corpus...')
    logging.info('number of samples is: {}'.format(len(training_samples)))

    collection = get_collection_data_structure(training_samples)
    se = SearchEngine("new-index").index(collection)

    # capture random ranking 
    random_ranking = []
    random.seed(42)
    for i in range(100):
        random_ranking.append({'index': random.randint(0, len(training_samples)), 'score': 0 })

    result = {}
    for ind, query in enumerate(queries):
        logging.info('indexing query: {}'.format(ind))
        ranking = se.search(query=query, return_docs=True, cutoff=100)
        new_ranking = []
        for item in ranking:
            new_ranking.append({'index': int(item['id']), 'score': round(float(item['score']), 2)})
        
        # sometimes due to small query, we get no rankings
        if len(new_ranking) == 0:
            result[ind] = random_ranking
        else:
            result[ind] = new_ranking
    
    return result
    
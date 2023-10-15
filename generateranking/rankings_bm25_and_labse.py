from utils import load_samples
import logging
import json
import torch
from sentence_transformers import SentenceTransformer, util


def get_labse_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset=False):            
    queries = load_samples(test_src_path)
    logging.info('loading corpus...')
    training_samples = load_samples(train_src_path)
    logging.info('loaded corpus...')
    logging.info('number of samples is: {}'.format(len(training_samples)))

    json_data = ''
    with open(bm25_file_name, 'r') as f:
        json_data = json.load(f)

    # compute embeddings for queries
    model = SentenceTransformer("sentence-transformers/LaBSE")
    with torch.no_grad():
        query_embeddings = model.encode(queries)

    result = {}
    for qid, query in enumerate(queries):
        logging.info('qid: {}'.format(qid))
        data = []
        bm25_rankings = json_data[str(qid)]
        bm25_rankings = list(map(lambda x: x["index"], bm25_rankings))
        bm25_sents = []
        for index in bm25_rankings:
            bm25_sents.append(training_samples[index])

        # compute embeddings for top 100 bm25 sentences
        with torch.no_grad():
            target_embeddings = model.encode(bm25_sents)
        scores = util.cos_sim(query_embeddings[qid], target_embeddings)
        scores = scores.numpy()
        scores = scores[0]

        # arrange indexes based on the LaBSE score
        ranking = []
        for (index, score) in zip(bm25_rankings, scores):
            ranking.append({"index": index, "score": round(float(score), 2)})
        
        if not is_ranking_for_devset:
            ranking.sort(key=lambda x: x['score'], reverse=True)
        result[qid] = ranking

    return result
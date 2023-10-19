from utils import load_samples
from utils.constants import ALL, QUERY_SRC, QUERY_DST, SRC_DST

import logging
import json
import torch
from sentence_transformers import SentenceTransformer, util


def get_3way_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset=False):
    queries = load_samples(test_src_path)
    logging.info('loading corpus...')
    training_src_samples = load_samples(train_src_path)
    training_dst_samples = load_samples(train_dst_path)
    logging.info('loaded corpus...')
    logging.info('number of samples is: {}'.format(len(training_src_samples)))

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
        bm25_src_sents = []
        bm25_dst_sents = []
        for index in bm25_rankings:
            bm25_src_sents.append(training_src_samples[index])
            bm25_dst_sents.append(training_dst_samples[index])

        # compute embeddings for top 100 bm25 sentences
        with torch.no_grad():
            target_src_embeddings = model.encode(bm25_src_sents)
            target_dst_embeddings = model.encode(bm25_dst_sents)

        scores_query_src = util.cos_sim(query_embeddings[qid], target_src_embeddings)
        scores_query_src = scores_query_src.numpy()
        scores_query_src = scores_query_src[0]

        scores_query_dst = util.cos_sim(query_embeddings[qid], target_dst_embeddings)
        scores_query_dst = scores_query_dst.numpy()
        scores_query_dst = scores_query_dst[0]

        scores_src_dst = util.cos_sim(target_src_embeddings, target_dst_embeddings)
        scores_src_dst = torch.diag(scores_src_dst, 0).tolist()

        # arrange indexes based on the LaBSE score
        ranking = []
        for (index, score_query_src, score_query_dst, score_src_dst) in zip(bm25_rankings, scores_query_src, scores_query_dst, scores_src_dst):
            ranking.append({"index": index, 
                            "labse_score_query_src": round(float(score_query_src), 2), 
                            "labse_score_query_dst": round(float(score_query_dst), 2),
                            "labse_score_src_dst": round(float(score_src_dst), 2),
                            "score": round(float(score_query_src) + float(score_query_dst) + float(score_src_dst), 2)
                            })
            
        if not is_ranking_for_devset:
            ranking.sort(key=lambda x: x['score'], reverse=True)
        result[qid] = ranking

    return result
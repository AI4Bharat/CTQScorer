from comet import download_model, load_from_checkpoint
from utils import load_samples
import logging
import json


def get_comet_qa_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset=False):
    queries = load_samples(test_src_path)
    logging.info('loading corpus...')
    training_src_examples = load_samples(train_src_path)
    training_dst_examples = load_samples(train_dst_path)
    logging.info('loaded corpus...')
    logging.info('number of samples is: {}'.format(len(training_src_examples)))

    json_data = ''
    with open(bm25_file_name, 'r') as f:
        json_data = json.load(f)
    
    # load comet qe model
    model_path = download_model("wmt20-comet-qe-da")
    model = load_from_checkpoint(model_path)

    result = {}
    for qid, query in enumerate(queries):
        logging.info('qid: {}'.format(qid))
        query_src_data = []
        query_dst_data = []
        src_dst_data = []
        
        bm25_rankings = json_data[str(qid)]
        bm25_rankings = list(map(lambda x: x["index"], bm25_rankings))
        for index in bm25_rankings:
            query_src_data.append({ "src": query, "mt": training_src_examples[index] })
            query_dst_data.append({ "src": query, "mt": training_dst_examples[index] })
            src_dst_data.append({ "src": training_src_examples[index], "mt": training_dst_examples[index] })
            
        model_output = model.predict(query_src_data, batch_size=100, gpus=1)
        query_src_scores = model_output[0]
        
        model_output = model.predict(query_dst_data, batch_size=100, gpus=1)
        query_dst_scores = model_output[0]
        
        model_output = model.predict(src_dst_data, batch_size=100, gpus=1)
        src_dst_scores = model_output[0]
        
        ranking = []
        for (index, query_src_score, query_dst_score, src_dst_score) in zip(bm25_rankings, query_src_scores, query_dst_scores, src_dst_scores):
            ranking.append({"index": index,
                            "comet_qe_query_src_score": round(float(query_src_score), 2),
                            "comet_qe_query_dst_score": round(float(query_dst_score), 2),
                            "comet_qe_src_dst_score": round(float(src_dst_score), 2)
                            })

        if not is_ranking_for_devset:
            ranking.sort(key=lambda x: x['comet_qe_query_src_score'], reverse=True)
        result[qid] = ranking

    return result

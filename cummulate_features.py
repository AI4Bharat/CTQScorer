from utils.commonutils import init_logging, load_samples, make_dir, get_json_data
from utils.utils_data import get_train_test_data
from utils.constants import *
import logging
import argparse

# constants
FEATURES_FOLDER = 'rankings_bm25_regression'
DATASET_TRAIN = 'dataset_train'
DATASET_TEST = 'dataset_test'


def get_recommendation_file_name(selection, training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset):
    path = '{}/recommendations_{}_{}_{}_{}.json'.format(selection, training_source, testing_source, src_lang, dst_lang)
    if is_ranking_for_devset:
        path = '{}/{}/recommendations_{}_{}_{}_{}.json'.format(FEATURES_FOLDER, selection, training_source, testing_source, src_lang, dst_lang)
    return path


def cummulate_feature_scores(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset):

    # get src and dst paths of parallel corpus along with queries path
    train_src_path, train_dst_path, test_src_path, test_dst_path = get_train_test_data(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset)
    logging.info('training src path: {}, dst path: {}, test src path: {}, test dst path: {}'.format(train_src_path, train_dst_path, test_src_path, test_dst_path))

    result = {}
    queries = load_samples(test_src_path)
    for qid, query in enumerate(queries):
        logging.info('qid: {}'.format(qid))
        
        result[qid] = {}

        # get max tokens 
        json_data = get_json_data(get_recommendation_file_name(RANKINGS_NO_OF_TOKENS, training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset))
        rankings = json_data[str(qid)]
        for ranking in rankings:
            index = ranking['index']
            # Initializatin is required only for the first time
            if index not in result[qid]:
                result[qid][index] = {}
            result[qid][index].update({
                'index': index,
                NO_OF_TOKENS_IN_QUERY : ranking['no_of_tokens_in_query'],
                NO_OF_TOKENS_IN_SRC_SENT : ranking['no_of_tokens_in_src_sent'],
                NO_OF_TOKENS_IN_DST_SENT : ranking['no_of_tokens_in_dst_sent']
            })

        # get LaBSE similarity score
        json_data = get_json_data(get_recommendation_file_name(RANKINGS_BM25_AND_3_WAY, training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset))
        rankings = json_data[str(qid)]
        for ranking in rankings:
            index = ranking['index']
            result[qid][index].update({
                LABSE_SCORE_QUERY_SRC : ranking['labse_score_query_src'],
                LABSE_SCORE_QUERY_DST : ranking['labse_score_query_dst'],
                LABSE_SCORE_SRC_DST : ranking['labse_score_src_dst']
            })

        # get CHRF score
        json_data = get_json_data(get_recommendation_file_name(RANKINGS_BM25_AND_CHRF, training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset))
        rankings = json_data[str(qid)]
        for ranking in rankings:
            index = ranking['index']
            result[qid][index].update({
                CHRF_SCORE: ranking['score'],
            })

        # get COMET QE score scores
        json_data = get_json_data(get_recommendation_file_name(RANKINGS_COMET_QA, training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset))
        rankings = json_data[str(qid)]
        for ranking in rankings:
            index = ranking['index']
            result[qid][index].update({
                COMET_QE_QUERY_SRC_SCORE : ranking['comet_qe_query_src_score'],
                COMET_QE_QUERY_DST_SCORE : ranking['comet_qe_query_dst_score'], 
                COMET_QE_SRC_DST_SCORE : ranking['comet_qe_src_dst_score']
            })
            
        # get perplexity scores
        json_data = get_json_data(get_recommendation_file_name(RANKINGS_BM25_AND_PERPLEXITY, training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset))
        rankings = json_data[str(qid)]
        for ranking in rankings:
            index = ranking['index']
            result[qid][index].update({
                SRC_PPL : ranking['src_ppl'],
                DST_PPL : ranking['dst_ppl'],
                SRC_DST_PPL : ranking['src_dst_ppl'],
                SRC_DST_QUERY_PPL : ranking['src_dst_query_ppl']
            })

    return result


def convert_to_csv(cummulated_features_json_data, ctqscorer_train_csv_file_name, ctqscorer_scores_file=None, is_ranking_for_devset=False):
    
    # Any new feature has to be added to this list
    features = [NO_OF_TOKENS_IN_QUERY, NO_OF_TOKENS_IN_SRC_SENT, NO_OF_TOKENS_IN_DST_SENT, 
                LABSE_SCORE_QUERY_SRC, LABSE_SCORE_QUERY_DST, LABSE_SCORE_SRC_DST,
                CHRF_SCORE, COMET_QE_QUERY_SRC_SCORE, COMET_QE_QUERY_DST_SCORE, COMET_QE_SRC_DST_SCORE,
                SRC_PPL, DST_PPL, SRC_DST_PPL, SRC_DST_QUERY_PPL]

    header = ['qid', 'index']
    header.extend(features)
    if is_ranking_for_devset and ctqscorer_scores_file:
        # 'qid_tmp' is same as 'qid'. 'index_tmp' is same as 'index'. Adding them to ensure they match when merging scores and rankings file.
        header.extend(['qid_tmp', 'index_tmp', 'comet_score', 'bleu_score', 'comet_qe_20_score', 'comet_da_22_score'])
        scores = []
        with open(ctqscorer_scores_file, 'r') as f:
            scores = f.read().splitlines()
        scores_index = 0
        
    with open(ctqscorer_train_csv_file_name, 'w') as f:
        f.write('{}\n'.format(','.join(header)))

    # iterate over each query id from the accumulated feature values
    for qid in list(cummulated_features_json_data.keys()):
        bm100 = cummulated_features_json_data[qid]

        # iterate over each bm25 elem from the pool of 100 bm25 samples.
        for index_of_elem_in_bm100 in list(bm100.keys()):
            elem = bm100[index_of_elem_in_bm100]
            row = []

            # add query and sample index
            row.append(qid)
            row.append(index_of_elem_in_bm100)
            
            # add all feature values
            for feature in features:
                row.append(elem[feature])

            if is_ranking_for_devset:
                # add row to training csv file
                with open(ctqscorer_train_csv_file_name, 'a') as f:
                    f.write('{},{}\n'.format(','.join(list(map(lambda x: str(x), row))), scores[scores_index]))
                    scores_index += 1
            else:
                # add row to training csv file
                with open(ctqscorer_train_csv_file_name, 'a') as f:
                    f.write('{}\n'.format(','.join(list(map(lambda x: str(x), row)))))


def main():
    """
    This program combines all the features into a single csv file. It can cummulate both train/test dataset using the flag `--devset`.
    """
    init_logging('cummulate_features.log')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="training source to be used")
    parser.add_argument("--test", help="testing source to be used")
    parser.add_argument("--src", help="source language")
    parser.add_argument("--dst", help="destination language")
    parser.add_argument("--devset", help="Is the ranking for devset?", action="store_true")
    args = parser.parse_args()

    training_source = args.train
    testing_source = args.test
    src_lang = args.src
    dst_lang = args.dst
    is_ranking_for_devset = True if args.devset else False
    
    make_dir(DATASET_TRAIN)
    make_dir(DATASET_TEST)

    if is_ranking_for_devset:
        ctqscorer_train_csv = '{}/{}_{}_{}.csv'.format(DATASET_TRAIN, training_source, src_lang, dst_lang)
    else:
        ctqscorer_train_csv = '{}/{}_{}_{}_{}.csv'.format(DATASET_TEST, training_source, testing_source, src_lang, dst_lang)
    
    cummulated_features_json_data_dict = cummulate_feature_scores(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset)
    
    if is_ranking_for_devset:
        one_shots_scores_file_name = 'outputs/regression_scores_{}_{}.csv'.format(src_lang, dst_lang)
        convert_to_csv(cummulated_features_json_data_dict, ctqscorer_train_csv_file_name=ctqscorer_train_csv, ctqscorer_scores_file=one_shots_scores_file_name, is_ranking_for_devset=is_ranking_for_devset)
    else:
        convert_to_csv(cummulated_features_json_data_dict, ctqscorer_train_csv_file_name=ctqscorer_train_csv)

if __name__ == '__main__':
    main()
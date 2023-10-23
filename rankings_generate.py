from utils import get_train_test_data, write_json_to_file, make_dir, init_logging
from generateranking import get_bm25_ranking, get_reranking_ranking, get_comet_qa_ranking, get_chrf_ranking, get_3way_ranking, get_ppl_ranking, get_no_of_tokens
from utils.constants import *

import os
import argparse
import logging


def generate_ranking(training_source, testing_source, src_lang, dst_lang, algorithm, is_ranking_for_devset=False, use_xglm_model=False):
    # get paths for data
    train_src_path, train_dst_path, test_src_path, test_dst_path = get_train_test_data(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset)
    logging.info('train_src_path: {}\ttrain_dst_path: {}\ttest_src_path: {}\ttest_dst_path: {}'.format(train_src_path, train_dst_path, test_src_path, test_dst_path))

    # other rankings use bm25 ranking, so generate a path for it 
    recommendations = 'recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang)
    bm25_file_name = 'rankings_bm25_regression/{}'.format(recommendations) if is_ranking_for_devset else '{}/rankings_bm25/{}'.format(EXAMPLE_SELECTION_TEST_DATA, recommendations)

    # compute ranking
    if algorithm == RANKINGS_BM25:
        result = get_bm25_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang)
        write_json_to_file(result, bm25_file_name)
    else:
        if not os.path.exists(bm25_file_name):
            logging.error('BM25 not exists: {}. Please refer to `rankings_generate.sh`.'.format(bm25_file_name))
            return

        if algorithm == RANKINGS_BM25_AND_RERANKING:
            result = get_reranking_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif algorithm == RANKINGS_COMET_QA:
            result = get_comet_qa_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif algorithm == RANKINGS_BM25_AND_CHRF:
            result = get_chrf_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif algorithm == RANKINGS_BM25_AND_3_WAY:
            result = get_3way_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif algorithm == RANKINGS_NO_OF_TOKENS:
            result = get_no_of_tokens(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif algorithm == RANKINGS_BM25_AND_PERPLEXITY:
            result = get_ppl_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset, use_xglm_model)
        else:
            logging.error('Invalid strategy: {}'.format(algorithm))
            return

        # write ranking to file
        ranking_file_name = 'rankings_bm25_regression/{}/{}'.format(algorithm, recommendations) if is_ranking_for_devset else '{}/{}/{}'.format(EXAMPLE_SELECTION_TEST_DATA, algorithm, recommendations)
        if is_ranking_for_devset:
            make_dir('rankings_bm25_regression/{}'.format(algorithm))
        else:
            make_dir('{}/{}'.format(EXAMPLE_SELECTION_TEST_DATA, algorithm))
        write_json_to_file(result, ranking_file_name)


def main():
    init_logging('generate_rankings.log')

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_src", help="training source to be used")
    parser.add_argument("--test_src", help="testing source to be used")
    parser.add_argument("--src_lang", help="source language")
    parser.add_argument("--dst_lang", help="destination language")
    parser.add_argument("--devset", help="Is the ranking for devset?", action="store_true")
    parser.add_argument("--algorithm", help="Algorithm for the ranking. use 'all' to compute all rankings")
    parser.add_argument("--xglm", help="Is the model used xglm?", action="store_true")
    args = parser.parse_args()
    
    logging.info(args)

    # args: training_source, queries_source, src_lang, dst_lang, is_ranking_for_devset
    training_source = args.train_src
    testing_source = args.test_src
    src_lang = args.src_lang
    dst_lang = args.dst_lang
    is_ranking_for_devset = True if args.devset else False
    algorithm = args.algorithm
    use_xglm_model = True if args.xglm else False

    if algorithm == 'all':
        for s in [RANKINGS_BM25, RANKINGS_BM25_AND_RERANKING, RANKINGS_BM25_AND_CHRF, RANKINGS_NO_OF_TOKENS,
                  RANKINGS_COMET_QA, RANKINGS_BM25_AND_3_WAY, RANKINGS_BM25_AND_PERPLEXITY]:
            generate_ranking(training_source, testing_source, src_lang, dst_lang, s, is_ranking_for_devset, use_xglm_model)

    else:
        generate_ranking(training_source, testing_source, src_lang, dst_lang, algorithm, is_ranking_for_devset, use_xglm_model)
    

# usage: python rankings_generate.py --train_src samanantar --test_src flores --src_lang hin_Deva --dst_lang eng_Latn --algorithm rankings_no_of_tokens
if __name__ == '__main__':
    main()
    
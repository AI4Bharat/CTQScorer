from utils import get_train_test_data, write_json_to_file, make_dir, init_logging
from generateranking import get_bm25_ranking, get_reranking_ranking, get_comet_qa_ranking, get_chrf_ranking, get_3way_ranking, get_ppl_ranking, get_no_of_tokens
from utils.constants import *

import os
import argparse
import logging


def generate_ranking(training_source, testing_source, src_lang, dst_lang, strategy, is_ranking_for_devset=False, use_xglm_model=False):
    # get paths for data
    train_src_path, train_dst_path, test_src_path, test_dst_path = get_train_test_data(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset)
    logging.info('train_src_path: {}\ttrain_dst_path: {}\ttest_src_path: {}\ttest_dst_path: {}'.format(train_src_path, train_dst_path, test_src_path, test_dst_path))

    # other rankings use bm25 ranking, so generate a path for it 
    recommendations = 'recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang)
    bm25_file_name = 'rankings_bm25_regression/{}'.format(recommendations) if is_ranking_for_devset else 'rankings_bm25/{}'.format(recommendations)

    # compute ranking
    if strategy == RANKINGS_BM25:
        result = get_bm25_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang)
        write_json_to_file(result, bm25_file_name)
    else:
        if not os.path.exists(bm25_file_name):
            logging.error('BM25 not exists: {}'.format(bm25_file_name))
            return

        if strategy == RANKINGS_BM25_AND_RERANKING:
            result = get_reranking_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif strategy == RANKINGS_COMET_QA:
            result = get_comet_qa_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif strategy == RANKINGS_BM25_AND_CHRF:
            result = get_chrf_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif strategy == RANKINGS_BM25_AND_3_WAY:
            result = get_3way_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif strategy == RANKINGS_NO_OF_TOKENS:
            result = get_no_of_tokens(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset)
        elif strategy == RANKINGS_BM25_AND_PERPLEXITY:
            result = get_ppl_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset, use_xglm_model)
        else:
            logging.error('Invalid strategy: {}'.format(strategy))
            return

        # write ranking to file
        ranking_file_name = 'rankings_bm25_regression/{}/{}'.format(strategy, recommendations) if is_ranking_for_devset else '{}/{}'.format(strategy, recommendations)
        if is_ranking_for_devset:
            make_dir('rankings_bm25_regression/{}'.format(strategy))
        else:
            make_dir(strategy)
        write_json_to_file(result, ranking_file_name)


def main():
    init_logging('generate_rankings.log')

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="training source to be used")
    parser.add_argument("--test", help="testing source to be used")
    parser.add_argument("--src", help="source language")
    parser.add_argument("--dst", help="destination language")
    parser.add_argument("--devset", help="Is the ranking for devset?", action="store_true")
    parser.add_argument("--strategy", help="strategy for the ranking. use 'all' to compute all rankings")
    parser.add_argument("--xglm", help="Is the model used xglm?", action="store_true")
    args = parser.parse_args()
    
    logging.info(args)

    # args: training_source, queries_source, src_lang, dst_lang, is_ranking_for_devset
    training_source = args.train
    testing_source = args.test
    src_lang = args.src
    dst_lang = args.dst
    is_ranking_for_devset = True if args.devset else False
    strategy = args.strategy
    use_xglm_model = True if args.xglm else False

    if strategy == 'all':
        for s in [RANKINGS_BM25, RANKINGS_BM25_AND_RERANKING, RANKINGS_BM25_AND_CHRF, RANKINGS_NO_OF_TOKENS,
                  RANKINGS_COMET_QA, RANKINGS_BM25_AND_3_WAY, RANKINGS_BM25_AND_PERPLEXITY]:
            generate_ranking(training_source, testing_source, src_lang, dst_lang, s, is_ranking_for_devset, use_xglm_model)

    else:
        generate_ranking(training_source, testing_source, src_lang, dst_lang, strategy, is_ranking_for_devset, use_xglm_model)
    

# usage: python rankings_generate.py --train samanantar --test flores --src hin_Deva --dst eng_Latn --strategy rankings_no_of_tokens
if __name__ == '__main__':
    main()
    
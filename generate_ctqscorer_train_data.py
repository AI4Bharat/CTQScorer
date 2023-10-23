import argparse
from tqdm.auto import tqdm
from sacrebleu import sentence_bleu

# utils
from core.model_parameters import model_parameters
from utils.constants import *
from utils.commonutils import make_dir, get_random_name, append_config_to_file, lang_abbr_to_lang, init_logging

# prompt construction
from core.prompts import get_n_shots, construct_prompt

# Preprocessing prompts, batching prompts and post processing outputs
from core.MTDataset import MTDataset
from core.process_outputs import predict_outputs

# scoring functions
from core.scoring_functions import init_comet_computation, init_comet_qe_20_computation, init_comet_da_22_computation
from core.scoring_functions import get_comet_scores, get_comet_qe_20_scores, get_comet_da_22_scores

# helper functions
from helper_functions import read_recommendations, get_samples, get_model


comet_da_20_metric = init_comet_computation()
comet_qe_20_metric = init_comet_qe_20_computation()
comet_da_22_metric = init_comet_da_22_computation()

"""
This function generates CTQ scores for (997 x 100 = 99700 tuples) which will be used as training data for the CTQScorer.
We have used FLORES-devset as Held-out Example Pairs and Samanantar/Europarl/Paracrawl as Example Datastore.
The CTQ scores are stored in 'outputs' directory where each file name is: regression_scores_{src_lang}_{dst_lang}.csv
For our experiments we consider, comet-da-20 score as CTQ scores. Outputs are also captured in the same directory.
"""
def get_ctq_scores(pipe, mp: model_parameters, experiment=''):
    model_name = mp.name.split('/')[1]
    
    # languages for which the model should be generate CTQ scores
    src_lang = lang_abbr_to_lang.get(mp.src_lang)
    dst_lang = lang_abbr_to_lang.get(mp.dst_lang)

    # create output directory
    output_dir, prompts_dir = 'outputs', 'prompts'
    make_dir(output_dir)
    make_dir(prompts_dir)

    # make note of configuration
    # '{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(name, type_of_algo, max_new_tokens, use_8_bit, toEnglish, num_of_shots, reranking, dataset, rec_source)
    scores_file = '{}/scores.csv'.format(output_dir)
    msg = '{} [{}]\n'.format(str(mp).strip(), experiment)
    append_config_to_file(scores_file, msg=msg)
    print(mp)

    # load samples from samanantar corpus
    src_train_samples, dst_train_samples, src_flores_dev_samples, dst_flores_dev_samples = get_samples(mp.training_source, mp.testing_source,
                                                                                        mp.src_lang, mp.dst_lang, is_ranking_for_devset=True)

    # get ranking of dev samples if reranking flag is true
    if mp.has_reranking:
        rankings = read_recommendations(mp.strategy, mp.training_source, mp.testing_source, mp.src_lang, mp.dst_lang, is_train_data=True)
        if len(rankings) == 0:
            print('No ranking found for: {}'.format(src_lang))

    # capture configuration and generate random name for file to map the configuration
    random_name = get_random_name()
    prediction_file = '{}/{}_{}_{}_{}_{}_shots_pred_{}.txt'.format(output_dir, experiment, model_name, src_lang, dst_lang, mp.no_of_shots, random_name)

    # write prompts to file
    with open('{}/{}_{}_{}.txt'.format(prompts_dir, experiment, mp.src_lang, mp.dst_lang), 'w') as f:
        f.write('')

    for qid, input_sample in enumerate(tqdm(src_flores_dev_samples)):

        # all prompts
        prompts = ''
            
        recommendations = []
        if mp.has_reranking:
            recommendations = rankings[str(qid)]

            # recommendation structure has been changed
            if mp.strategy == RANKINGS_BM25_REGRESSION:
                # recommendations are in [{ "index": 630729, "score": 37.21}, ... ]
                recommendations = list(map(lambda x: x["index"], recommendations))

        # create an object to batch the examples
        datasetObj = MTDataset()

        for recommendation in recommendations:

            # prompt construction
            shots = get_n_shots(mp, src_train_samples, dst_train_samples, mp.no_of_shots, src_lang, dst_lang, recommendations=[recommendation])
            content = construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=1)
            prompts = prompts + '{}\n{}\n\n\n'.format(qid, content)

            # print(content)
            # print('\n\n\n')
            datasetObj.addprompt(content)
            datasetObj.addinput(input_sample)
    
        # write prompts to file for reference
        with open('{}/{}_{}_{}.txt'.format(prompts_dir, experiment, mp.src_lang, mp.dst_lang), 'a') as f:
            f.write(prompts)

        # obtained the output from model
        pred_dst = predict_outputs(pipe, datasetObj, prediction_file, mp.name) 
        # print(pred_dst)

        # obtain comet score
        refs = [dst_flores_dev_samples[qid]] * len(pred_dst)
        srcs = [src_flores_dev_samples[qid]] * len(pred_dst)
        # print(refs)
        # print(srcs)
        comet_scores = get_comet_scores(predicted=pred_dst, references=refs, source=srcs, comet_da_20_metric=comet_da_20_metric)
        comet_scores = list(map(lambda x: round(x, 4), comet_scores))
        # print('COMET score -> {}'.format(comet_scores))

        bleu_scores = []
        ref = dst_flores_dev_samples[qid]
        for candidate in pred_dst:
            bleu_scores.append(sentence_bleu(candidate, [ref]).score)
        bleu_scores = list(map(lambda x: round(x, 2), bleu_scores))
        # print('BLEU scores -> {}'.format(bleu_scores))
        
        comet_qe_20_scores = get_comet_qe_20_scores(predicted=pred_dst, source=srcs)
        comet_qe_20_scores = list(map(lambda x: round(x, 4), comet_qe_20_scores))

        comet_da_22_scores = get_comet_da_22_scores(predicted=pred_dst, references=refs, source=srcs)
        comet_da_22_scores = list(map(lambda x: round(x, 4), comet_da_22_scores))

        # write scores to regression scores file
        regression_scores_file = '{}/regression_scores_{}_{}.csv'.format(output_dir, mp.src_lang, mp.dst_lang)
        for elem_id_in_corpus, comet_score, bleu_score, comet_qe_20_score, comet_da_22_score in zip(recommendations, comet_scores, bleu_scores, comet_qe_20_scores, comet_da_22_scores):
            with open(regression_scores_file, 'a') as f:
                f.write('{},{},{},{},{},{}\n'.format(qid, elem_id_in_corpus, comet_score, bleu_score, comet_qe_20_score, comet_da_22_score))
                

def main():
    init_logging('generate_ctqscorer_train_data.log')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="training source to be used")
    parser.add_argument("--test", help="testing source to be used")
    parser.add_argument("--src", help="source language")
    parser.add_argument("--dst", help="destination language")
    parser.add_argument("--xglm", help="Is the model used xglm?", action="store_true")
    args = parser.parse_args()

    name = XGLM_7B if args.xglm else BLOOM_7B
    mp = model_parameters(name=name)

    mp.training_source = args.train
    mp.testing_source = args.test
    mp.src_lang = args.src
    mp.dst_lang = args.dst
    mp.strategy = RANKINGS_BM25_REGRESSION
    mp.has_reranking = True

    # generate pipe and use the same pipe
    pipe = get_model(mp.name, type_of_algo=mp.type_of_algo, use_8_bit=mp.use_8_bit)
    get_ctq_scores(pipe, mp, experiment='')

if __name__ == '__main__':
    main()
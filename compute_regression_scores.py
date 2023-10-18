import argparse
from model_parameters import model_parameters
from utils_language import *
from utils import *
from tqdm.auto import tqdm
# from bloom import get_samples, read_recommendations
# from bloom import get_comet_scores, get_comet_qe_20_scores, get_comet_da_22_scores
from constants import *
from sacrebleu import sentence_bleu, corpus_bleu

def read_recommendations(strategy, training_source, testing_source, src_lang, dst_lang, strategy_nested=''):
    json_data = {}

    if strategy_nested == '':
        recommendations = 'recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang)
        ranking_file_name = '{}/{}'.format(strategy, recommendations)
        with open(ranking_file_name, 'r') as f:
            json_data = json.load(f)
    else:
        recommendations = 'recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang)
        ranking_file_name = '{}/{}/{}'.format(strategy, strategy_nested, recommendations)
        with open(ranking_file_name, 'r') as f:
            json_data = json.load(f)
    
    return json_data

from utils_data import get_train_test_data
def get_samples(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset=False, intr_lang=''):
    train_src_path, train_dst_path, test_src_path, test_dst_path = get_train_test_data(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset)
    src_train_samples = load_samples(train_src_path)
    dst_train_samples = load_samples(train_dst_path)
    src_test_samples = load_samples(test_src_path)
    dst_test_samples = load_samples(test_dst_path)

    # consider intermediate language in case
    if intr_lang != '':
        if training_source == 'flores':
            intr_train_path = 'dataset/train/{}.dev'.format(intr_lang)
        elif training_source == 'samanantar':
            intr_lang = lang_abbr_to_lang_code.get(intr_lang)
            intr_train_path = 'dataset/samanantar/en-{}/train.{}'.format(intr_lang, intr_lang)
        intr_lang_samples = load_samples(intr_train_path)
        return src_train_samples, dst_train_samples, src_test_samples, dst_test_samples, intr_lang_samples

    return src_train_samples, dst_train_samples, src_test_samples, dst_test_samples


from comet import download_model, load_from_checkpoint
def init_comet_computation():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # comet_metric = load('comet' , 'Unbabel/wmt20-comet-da')
    
    model_path = download_model("Unbabel/wmt20-comet-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric

def init_comet_qe_20_computation():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # comet_metric = load('comet' , 'Unbabel/wmt20-comet-da')
    
    model_path = download_model("Unbabel/wmt20-comet-qe-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric

def init_comet_da_22_computation():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # comet_metric = load('comet' , 'Unbabel/wmt20-comet-da')
    
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric

comet_da_20_metric = init_comet_computation()
def get_comet_scores(predicted, references, source):
    comet_metric = comet_da_20_metric
    # comet_metric = load('comet')
    scores = []

    # sometimes we just run for 5 to 10 samples
    k = len(predicted)
    references = references[:k]
    source = source[:k]

    idx = 0
    while idx < len(predicted):
        batch = int(min(1024, len(predicted) - idx))
        predicted_batch = predicted[idx: idx + batch]
        references_batch = references[idx: idx + batch]
        source_batch = source[idx: idx + batch]

        data = []
        for src, mt, ref in zip(source_batch, predicted_batch, references_batch):
            data.append({
                "src": src,
                "mt": mt,
                "ref": ref
            })
        
        comet_score = comet_metric.predict(data, progress_bar=True)        
        # comet_score = comet_metric.compute(predictions=predicted_batch, references=references_batch, sources=source_batch, progress_bar=True)
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores

comet_qe_20_metric = init_comet_qe_20_computation()
def get_comet_qe_20_scores(predicted, source):
    comet_metric = comet_qe_20_metric
    scores = []

    # sometimes we just run for 5 to 10 samples
    k = len(predicted)
    source = source[:k]

    idx = 0
    while idx < len(predicted):
        batch = int(min(1024, len(predicted) - idx))
        predicted_batch = predicted[idx: idx + batch]
        source_batch = source[idx: idx + batch]

        data = []
        for src, mt in zip(source_batch, predicted_batch):
            data.append({
                "src": src,
                "mt": mt,
            })
        
        comet_score = comet_metric.predict(data, progress_bar=True)        
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores

comet_da_22_metric = init_comet_da_22_computation()
def get_comet_da_22_scores(predicted, references, source):
    comet_metric = comet_da_22_metric
    scores = []

    # sometimes we just run for 5 to 10 samples
    k = len(predicted)
    references = references[:k]
    source = source[:k]

    idx = 0
    while idx < len(predicted):
        batch = int(min(1024, len(predicted) - idx))
        predicted_batch = predicted[idx: idx + batch]
        references_batch = references[idx: idx + batch]
        source_batch = source[idx: idx + batch]

        data = []
        for src, mt, ref in zip(source_batch, predicted_batch, references_batch):
            data.append({
                "src": src,
                "mt": mt,
                "ref": ref
            })
        
        comet_score = comet_metric.predict(data, progress_bar=True)        
        # comet_score = comet_metric.compute(predictions=predicted_batch, references=references_batch, sources=source_batch, progress_bar=True)
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores



# This function evaluates the BLOOM model and also captures the MT outputs
def get_missing_comet_scores(mp: model_parameters, output_file):    
    # languages for which the model should be evaluated
    src_lang = lang_abbr_to_lang.get(mp.src_lang)
    dst_lang = lang_abbr_to_lang.get(mp.dst_lang)

    output_dir = 'outputs'

    # load samples from samanantar corpus
    src_train_samples, dst_train_samples, src_flores_dev_samples, dst_flores_dev_samples = get_samples(mp.training_source, mp.testing_source,
                                                                                        mp.src_lang, mp.dst_lang, is_ranking_for_devset=True)
    
    # get ranking of dev samples if reranking flag is true
    if mp.has_reranking:
        rankings = read_recommendations(mp.strategy, mp.training_source, mp.testing_source, mp.src_lang, mp.dst_lang)
        if len(rankings) == 0:
            print('No ranking found for: {}'.format(src_lang))


    outputs = ''
    with open(output_file, 'r') as f:
        outputs = f.read()

    outputs = outputs.splitlines()
    current_index = 0
    batch = 100

    # write scores to regression file
    regression_scores_file = '{}/regression_scores_{}_{}.csv'.format(output_dir, mp.src_lang, mp.dst_lang)
    with open(regression_scores_file, 'w') as f:
        f.write('')
        
    for qid, input_sample in enumerate(tqdm(src_flores_dev_samples)):

        recommendations = []
        if mp.has_reranking:
            recommendations = rankings[str(qid)]

            # recommendation structure has been changed
            if mp.strategy == RANKINGS_BM25_REGRESSION:
                # recommendations are in [{ "index": 630729, "score": 37.21}, ... ]
                recommendations = list(map(lambda x: x["index"], recommendations))


        # obtained the output from model
        pred_dst = outputs[current_index: current_index + batch]
        current_index = current_index + batch
        # print(pred_dst)

        # obtain comet score
        refs = [dst_flores_dev_samples[qid]] * len(pred_dst)
        srcs = [src_flores_dev_samples[qid]] * len(pred_dst)

        comet_scores = get_comet_scores(predicted=pred_dst, references=refs, source=srcs)
        comet_scores = list(map(lambda x: round(x, 4), comet_scores))
        # print('COMET score -> {}'.format(comet_scores))

        bleu_scores = []
        ref = dst_flores_dev_samples[qid]
        for candidate in pred_dst:
            bleu_scores.append(sentence_bleu(candidate, [ref]).score)
        bleu_scores = list(map(lambda x: round(x, 2), bleu_scores))


        comet_qe_20_scores = get_comet_qe_20_scores(predicted=pred_dst, source=srcs)
        comet_qe_20_scores = list(map(lambda x: round(x, 4), comet_qe_20_scores))

        comet_da_22_scores = get_comet_da_22_scores(predicted=pred_dst, references=refs, source=srcs)
        comet_da_22_scores = list(map(lambda x: round(x, 4), comet_da_22_scores))

        # print('BLEU scores -> {}'.format(bleu_scores))
        for elem_id_in_corpus, comet_score, bleu_score, comet_qe_20_score, comet_da_22_score in zip(recommendations, comet_scores, bleu_scores, comet_qe_20_scores, comet_da_22_scores):
            with open(regression_scores_file, 'a') as f:
                f.write('{},{},{},{},{},{}\n'.format(qid, elem_id_in_corpus, comet_score, bleu_score, comet_qe_20_score, comet_da_22_score))


def main():
    init_logging('compute_regression_scores.log')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="training source to be used")
    parser.add_argument("--test", help="testing source to be used")
    parser.add_argument("--src", help="source language")
    parser.add_argument("--dst", help="destination language")
    parser.add_argument("--outputs", help="location of outputs file")
    args = parser.parse_args()

    name = BLOOM_7B
    mp = model_parameters(name=name)

    mp.training_source = args.train
    mp.testing_source = args.test
    mp.src_lang = args.src
    mp.dst_lang = args.dst
    mp.strategy = RANKINGS_BM25_REGRESSION
    mp.has_reranking = True

    get_missing_comet_scores(mp, args.outputs)

if __name__ == '__main__':
    main()
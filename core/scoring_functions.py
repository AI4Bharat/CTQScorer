import os
import numpy as np
from comet import download_model, load_from_checkpoint
from evaluate import load


def init_chrf():
    chrf = load("chrf")
    return chrf

# Fix to get around torch error for computing comet score
def init_comet_computation():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"    
    model_path = download_model("Unbabel/wmt20-comet-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric

def init_comet_qe_20_computation():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    model_path = download_model("Unbabel/wmt20-comet-qe-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric

def init_comet_da_22_computation():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_metric = load_from_checkpoint(model_path)
    return comet_metric


def get_chrf_scores(predicted, references, chrf):
    tmp_references = []
    for reference in references:
        tmp_references.append([reference])
    references = tmp_references

    chrfscore = chrf.compute(predictions=predicted, references=references).get('score')
    chrfpp_score = chrf.compute(predictions=predicted, references=references, word_order=2).get('score')
    chrfscore = round(chrfscore, 2)
    chrfpp_score = round(chrfpp_score, 2)
    return chrfscore, chrfpp_score

def get_comet_scores(predicted, references, source, comet_da_20_metric):
    comet_metric = comet_da_20_metric
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
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores

def get_comet_mean_score(predicted, references, source, comet_da_20_metric):
    scores = get_comet_scores(predicted, references, source, comet_da_20_metric)
    mean_score = np.mean(scores)
    mean_score = round(mean_score, 4)
    return mean_score

def get_comet_qe_20_scores(predicted, source, comet_qe_20_metric):
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

def get_comet_da_22_scores(predicted, references, source, comet_da_22_metric):
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
        scores.extend(comet_score['scores'])
        idx += batch
    
    return scores
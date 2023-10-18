# %%
# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
import sys
import json
from utils import make_dir
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# %%
from constants import *

# CONSTANTS
COMET_SCORE = 'comet_score'
BLEU_SCORE = 'bleu_score'
NO_OF_TOKENS_IN_QUERY = 'no_of_tokens_in_query'
NO_OF_TOKENS_IN_SRC_SENT = 'no_of_tokens_in_src_sent'
NO_OF_TOKENS_IN_DST_SENT = 'no_of_tokens_in_dst_sent'
LABSE_SCORE_QUERY_SRC = 'labse_score_query_src'
LABSE_SCORE_QUERY_DST = 'labse_score_query_dst'
LABSE_SCORE_SRC_DST = 'labse_score_src_dst'
CHRF_SCORE = 'chrf_score'

COMET_QE_QUERY_SRC_SCORE = 'comet_qe_query_src_score'
COMET_QE_QUERY_DST_SCORE = 'comet_qe_query_dst_score'
COMET_QE_SRC_DST_SCORE = 'comet_qe_src_dst_score'
SRC_DST_PPL = 'src_dst_ppl'
SRC_DST_QUERY_PPL = 'src_dst_query_ppl'

IN22_OTHER_SOURCES = 'in22_other_sources'
SAMANANTAR = 'samanantar'
FLORES = 'flores'

SAVED_MODELS = 'saved_models'
DATASET_TRAIN = 'dataset_train'
DATASET_TEST = 'dataset_test'

TOKEN_RATIO_QUERY_TO_SRC = 'token_ratio_query_to_src'
TOKEN_RATIO_SRC_TO_DST = 'token_ratio_src_to_dst'

def job(training_source, testing_source, src_lang, dst_lang):
    # %%
    # features = [NO_OF_TOKENS_IN_QUERY, NO_OF_TOKENS_IN_SRC_SENT, NO_OF_TOKENS_IN_DST_SENT,
    #             LABSE_SCORE_QUERY_SRC, LABSE_SCORE_QUERY_DST, LABSE_SCORE_SRC_DST,
    #             CHRF_SCORE, 
    #             COMET_QE_QUERY_SRC_SCORE, COMET_QE_QUERY_DST_SCORE, COMET_QE_QUERY_DST_SCORE,
    #             SRC_DST_PPL, SRC_DST_QUERY_PPL,
    #             TOKEN_RATIO_QUERY_TO_SRC, TOKEN_RATIO_SRC_TO_DST] 

    features = [NO_OF_TOKENS_IN_QUERY, NO_OF_TOKENS_IN_SRC_SENT, NO_OF_TOKENS_IN_DST_SENT,
                LABSE_SCORE_QUERY_SRC, LABSE_SCORE_QUERY_DST, LABSE_SCORE_SRC_DST,
                CHRF_SCORE, 
                COMET_QE_QUERY_SRC_SCORE, COMET_QE_QUERY_DST_SCORE, COMET_QE_QUERY_DST_SCORE,
                SRC_DST_PPL, SRC_DST_QUERY_PPL] 

    # NO_OF_TOKENS_IN_QUERY, NO_OF_TOKENS_IN_SRC_SENT, NO_OF_TOKENS_IN_DST_SENT,
                    # SRC_DST_PPL, SRC_DST_QUERY_PPL]
                    # COMET_QE_QUERY_SRC_SCORE, COMET_QE_QUERY_DST_SCORE, COMET_QE_QUERY_DST_SCORE, COMET_QE_SRC_DST_SCORE, SRC_DST_PPL, SRC_DST_QUERY_PPL]

    # %% [markdown]
    # Features = [LABSE_SCORE_QUERY_SRC, LABSE_SCORE_QUERY_DST, LABSE_SCORE_SRC_DST,
    #             CHRF_SCORE, 
    #             COMET_QE_QUERY_SRC_SCORE, COMET_QE_QUERY_DST_SCORE, COMET_QE_QUERY_DST_SCORE,
    #             SRC_DST_PPL, SRC_DST_QUERY_PPL] 
    # 
    # Filter negative comet scores.
    # 
    # Increase train dataset;
    # 
    # Ignore number of tokens;
    # 
    # All languages; All directions;

    # %% [markdown]
    # ### Inputs

    # %%
    # training_source = EUROPARL
    # testing_source = FLORES
    # src_lang = DEU_LATN
    # dst_lang = ENG_LATN

    # %% [markdown]
    # ### Load and process data

    # %%
    # Training dataset is created using bloom.ipynb file. (Refer to get_prompt_scores function).
    dataset_path = '{}/{}_{}_{}.csv'.format(DATASET_TRAIN, training_source, src_lang, dst_lang)
    dataset = pd.read_csv(dataset_path)
    dataset.replace([np.inf], 99999, inplace=True)
    # dataset = dataset[dataset[COMET_SCORE] >= 0]
    # dataset[TOKEN_RATIO_QUERY_TO_SRC] = dataset[NO_OF_TOKENS_IN_QUERY].div(dataset[NO_OF_TOKENS_IN_SRC_SENT], fill_value=1)
    # dataset[TOKEN_RATIO_SRC_TO_DST] = dataset[NO_OF_TOKENS_IN_SRC_SENT].div(dataset[NO_OF_TOKENS_IN_DST_SENT], fill_value=1)
    dataset.replace([np.inf], 0, inplace=True)
    dataset = dataset.drop(['qid_tmp', 'index_tmp'], axis=1)
    dataset

    # %%
    # create feature variables and y
    df = dataset.copy()
    X = df.drop(['comet_score', 'bleu_score'], axis=1)
    y = df[['comet_score']]

    # create train/val/test dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=10)
    # X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=10)

    # %%
    # pick only the necessary features
    X_train = X_train[features]
    X_val = X_val[features]
    # X_test = X_test[features]

    # %%
    X_train

    # %%
    # Standardizing data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)

    y_scalar = MinMaxScaler()
    y_scalar.fit(y_train)
    y_train = y_scalar.transform(y_train)
    y_val = y_scalar.transform(y_val)
    # y_test = y_scalar.transform(y_test)

    # %%
    len(features) == len(X_val[0])

    # %%
    # set seed
    def set_seed():
        torch.manual_seed(42)
        np.random.seed(42)

    set_seed()

    # %%
    """
    NO_OF_TOKENS_IN_QUERY = 'no_of_tokens_in_query'
    NO_OF_TOKENS_IN_SRC_SENT = 'no_of_tokens_in_src_sent'
    NO_OF_TOKENS_IN_DST_SENT = 'no_of_tokens_in_dst_sent'
    LABSE_SCORE_QUERY_SRC = 'labse_score_query_src'
    LABSE_SCORE_QUERY_DST = 'labse_score_query_dst'
    LABSE_SCORE_SRC_DST = 'labse_score_src_dst'
    CHRF_SCORE = 'chrf_score'
    COMET_QA_SCORE = 'comet_qa_score'
    """
    # visualize the data
    # sns.scatterplot(x=COMET_QA_SCORE, y=COMET_SCORE, data=dataset)

    # %%
    # # features = [NO_OF_TOKENS_IN_QUERY, NO_OF_TOKENS_IN_SRC_SENT, NO_OF_TOKENS_IN_DST_SENT, 
    # #                 LABSE_SCORE_QUERY_SRC, LABSE_SCORE_QUERY_DST, LABSE_SCORE_SRC_DST,
    # #                 CHRF_SCORE, COMET_QA_SCORE ]

    # from sklearn.preprocessing import StandardScaler, MinMaxScaler 
    # scaled_features =  X_train[features]
    # x_scalar = MinMaxScaler().fit(scaled_features.values)
    # scaled_features = x_scalar.transform(scaled_features.values)
    # X_train[features] = scaled_features
    # X_val[features] = x_scalar.transform(X_val[features].values)
    # X_test[features] = x_scalar.transform(X_test[features].values)

    # scaled_features = y_train[[COMET_SCORE]]
    # y_scalar = MinMaxScaler().fit(scaled_features.values)
    # scaled_features = y_scalar.transform(scaled_features.values)
    # y_train[COMET_SCORE] = scaled_features
    # y_val[COMET_SCORE] = y_scalar.transform(y_val[[COMET_SCORE]].values)
    # y_test[COMET_SCORE] = y_scalar.transform(y_test[[COMET_SCORE]].values)

    # %%
    # create a regression model
    model = LinearRegression()

    # define and train the Ridge model with regularization parameter alpha
    # from sklearn.linear_model import Ridge
    # alpha = 0.2
    # model = Ridge(alpha=alpha)
    # ridge.fit(X_train, y_train)

    # fit the model
    model.fit(X_train, y_train)

    # make predictions
    y_val_pred = model.predict(X_val)

    # model evaluation
    print('mean_squared_error : ', mean_squared_error(y_val, y_val_pred))
    print('mean_absolute_error : ', mean_absolute_error(y_val, y_val_pred))

    # %%
    # mean_squared_error :  0.027887389493700655
    # mean_absolute_error :  0.11378171350541372

    # %%
    # TODO: Didn't use test data from the training dataset

    # %% [markdown]
    # ### Get prompt score for FLORES devtest

    # %%
    # load the test dataset
    test_data_path = '{}/{}_{}_{}_{}.csv'.format(DATASET_TEST, training_source, testing_source, src_lang, dst_lang)
    X_test = pd.read_csv(test_data_path)
    X_test.replace([np.inf], 99999, inplace=True)
    # X_test[TOKEN_RATIO_QUERY_TO_SRC] = X_test[NO_OF_TOKENS_IN_QUERY].div(X_test[NO_OF_TOKENS_IN_SRC_SENT], fill_value=1)
    # X_test[TOKEN_RATIO_SRC_TO_DST] = X_test[NO_OF_TOKENS_IN_SRC_SENT].div(X_test[NO_OF_TOKENS_IN_DST_SENT], fill_value=1)
    # X_test.replace([np.inf], 0, inplace=True)
    
    scaled_features =  X_test[features]
    scaled_features = scaler.transform(scaled_features.values)
    X_test[features] = scaled_features
    X_test

    # %%
    # predict for test data
    y_test_pred = model.predict(X_test[features])
    X_test['prompt_score'] = y_test_pred
    X_test

    # %%
    # clean the test outputs
    X_test['prompt_score'] = X_test['prompt_score'].apply(lambda x: round(x, 4))
    X_test = X_test.sort_values(by=['qid'])

    # sort the predicted scores 
    result = {}
    for i, row in X_test.iterrows():
        qid, index, pred_comet_score = row['qid'], row['index'], row['prompt_score']
        # print(qid, index, pred_comet_score)
        qid, index = int(qid), int(index)
        if qid not in result:
            result[qid] = []
        
        result[qid].append({"index": index, "score": pred_comet_score})

    # sort based on the predicted prompt score
    for qid in list(result.keys()):
        ranking = result[qid]
        ranking.sort(key=lambda x: x['score'], reverse=True)
        result[qid] = ranking

    # write score to a JSON file
    make_dir('rankings_regression')
    with open('rankings_regression/recommendations_{}_{}_{}_{}.json'.format(training_source, testing_source, src_lang, dst_lang), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


# job(training_source=SAMANANTAR, testing_source=FLORES, src_lang=BEN_BENG, dst_lang=ENG_LATN)
# job(training_source=SAMANANTAR, testing_source=FLORES, src_lang=GUJ_GUJR, dst_lang=ENG_LATN)
job(training_source=SAMANANTAR, testing_source=FLORES, src_lang=HIN_DEVA, dst_lang=ENG_LATN)
# job(training_source=SAMANANTAR, testing_source=FLORES, src_lang=ENG_LATN, dst_lang=BEN_BENG)
# job(training_source=SAMANANTAR, testing_source=FLORES, src_lang=ENG_LATN, dst_lang=GUJ_GUJR)
# job(training_source=SAMANANTAR, testing_source=FLORES, src_lang=ENG_LATN, dst_lang=HIN_DEVA)



# job(training_source=EUROPARL, testing_source=FLORES, src_lang=FRA_LATN, dst_lang=ENG_LATN)
# job(training_source=EUROPARL, testing_source=FLORES, src_lang=DEU_LATN, dst_lang=ENG_LATN)
# job(training_source=PARACRAWL, testing_source=FLORES, src_lang=RUS_CYRL, dst_lang=ENG_LATN)
# job(training_source=EUROPARL, testing_source=FLORES, src_lang=ENG_LATN, dst_lang=FRA_LATN)
# job(training_source=EUROPARL, testing_source=FLORES, src_lang=ENG_LATN, dst_lang=DEU_LATN)
# job(training_source=PARACRAWL, testing_source=FLORES, src_lang=ENG_LATN, dst_lang=RUS_CYRL)
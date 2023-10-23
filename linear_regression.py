# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import json

from utils import make_dir
from utils.constants import *
from utils.commonutils import set_seed

# %% [markdown]
# ### Load and process data

# %%
def data_preprocessing(training_source, src_lang, dst_lang, features):    
    # CTQScorer training dataset should be created before training linear regression model. 
    dataset_path = '{}/{}_{}_{}.csv'.format(DATASET_TRAIN, training_source, src_lang, dst_lang)
    dataset = pd.read_csv(dataset_path)
    
    # data cleaning
    dataset = dataset.drop(['qid_tmp', 'index_tmp'], axis=1)
    dataset.replace([np.inf], 99999, inplace=True)

    # create feature variables and use target as comet_score; Any new metric can be easily incorporated here.
    df = dataset.copy()
    X = df.drop(['comet_score', 'bleu_score', 'comet_qe_20_score', 'comet_da_22_score'], axis=1)
    y = df[['comet_score']]

    # create train/val dataset
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.2, random_state=10)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=10)

    # pick only the features we wish to train on 
    X_train = X_train[features]
    X_val = X_val[features]

    # Standardizing data
    X_scalar = MinMaxScaler()
    X_scalar.fit(X_train)
    X_train = X_scalar.transform(X_train)
    X_val = X_scalar.transform(X_val)

    y_scalar = MinMaxScaler()
    y_scalar.fit(y_train)
    y_train = y_scalar.transform(y_train)
    y_val = y_scalar.transform(y_val)
    
    return X_train, y_train, X_val, y_val, X_scalar

# %% [markdown]
# ### Train linear regression model

# %%
def train_model(X_train, y_train, X_val, y_val):
    set_seed()

    # create a regression model
    model = LinearRegression()

    # fit the model
    model.fit(X_train, y_train)

    # make predictions
    y_val_pred = model.predict(X_val)

    # model evaluation
    print('Validation mean_squared_error : ', mean_squared_error(y_val, y_val_pred))
    print('Validation mean_absolute_error : ', mean_absolute_error(y_val, y_val_pred))
    
    return model

# %% [markdown]
# ### Get salient feature importance

# %%
def get_importance_coeff(model, columns):
    importances = pd.DataFrame(data={
        'Attribute': columns,
        'Importance': model.coef_[0]
    })
    importances = importances.sort_values(by='Importance', ascending=False)
    importances.to_csv('importances.csv')
    
    importance_coefficient = model.coef_[0]
    for item1, item2 in zip(columns, importance_coefficient):
        print('{}, {}'.format(item1, round(item2, 3)))

    plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    plt.title('Feature importances obtained from coefficients', size=20)
    plt.xticks(rotation='vertical')
    plt.show()

# %% [markdown]
# ### Generate CTQScorer ranking using linear regression for test data

# %%
def generate_ctqscorer_ranking(training_source, testing_source, src_lang, dst_lang, x_scalar, features, model):
    # load the test dataset
    test_data_path = '{}/{}_{}_{}_{}.csv'.format(DATASET_TEST, training_source, testing_source, src_lang, dst_lang)
    X_test = pd.read_csv(test_data_path) 
    X_test.replace([np.inf], 99999, inplace=True)

    # predict ctq score
    scaled_features =  X_test[features]
    scaled_features = x_scalar.transform(scaled_features.values)
    X_test[features] = scaled_features
    ctq_scores = model.predict(X_test[features])
    X_test['ctq_score'] = ctq_scores

    # clean the test outputs
    X_test['ctq_score'] = X_test['ctq_score'].apply(lambda x: round(x, 4))
    X_test = X_test.sort_values(by=['qid'])

    # sort the predicted scores 
    result = {}
    for i, row in X_test.iterrows():
        qid, index, ctq_score = row['qid'], row['index'], row['ctq_score']
        qid, index = int(qid), int(index)
        if qid not in result:
            result[qid] = []
        
        result[qid].append({"index": index, "score": ctq_score})

    # sort based on the predicted prompt score
    for qid in list(result.keys()):
        ranking = result[qid]
        ranking.sort(key=lambda x: x['score'], reverse=True)
        result[qid] = ranking

    # write score to a JSON file
    make_dir('{}/{}'.format(EXAMPLE_SELECTION_TEST_DATA, RANKINGS_LINEAR_REGRESSION))
    with open('{}/{}/recommendations_{}_{}_{}_{}.json'.format(EXAMPLE_SELECTION_TEST_DATA, RANKINGS_LINEAR_REGRESSION, training_source, testing_source, src_lang, dst_lang), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

# %% [markdown]
# ### Main function

# %%
training_source = SAMANANTAR
testing_source = FLORES
src_lang = HIN_DEVA
dst_lang = ENG_LATN

# We excluded SRC_PPL, DST_PPL features. Any new feature must be incorporated here
features = [NO_OF_TOKENS_IN_QUERY, 
            NO_OF_TOKENS_IN_SRC_SENT, 
            NO_OF_TOKENS_IN_DST_SENT,
            LABSE_SCORE_QUERY_SRC, 
            LABSE_SCORE_QUERY_DST, 
            LABSE_SCORE_SRC_DST,
            CHRF_SCORE, 
            COMET_QE_QUERY_SRC_SCORE, 
            COMET_QE_QUERY_DST_SCORE, 
            COMET_QE_SRC_DST_SCORE,
            SRC_DST_PPL, 
            SRC_DST_QUERY_PPL] 

# %%
X_train, y_train, X_val, y_val, X_scalar = data_preprocessing(training_source, src_lang, dst_lang, features)
model = train_model(X_train, y_train, X_val, y_val)
generate_ctqscorer_ranking(training_source, testing_source, src_lang, dst_lang, X_scalar, features, model)
get_importance_coeff(model, features)



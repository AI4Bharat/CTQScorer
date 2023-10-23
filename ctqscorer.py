# %%
import torch
import torch.nn as nn
import tqdm
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import os
import json
import copy
import numpy as np
import pandas as pd
import functools
import argparse

from utils.commonutils import make_dir, set_seed, init_logging
from utils.constants import *
from core.NeuralNet import NeuralNet, get_activation_func, get_optimizer

# %% [markdown]
# ### Load and process data

# %%
def data_preprocessing(training_source, src_lang, dst_lang, features, device):
    
    # CTQScorer training dataset should be created before training CTQScorer. 
    dataset_path = '{}/{}_{}_{}.csv'.format(DATASET_TRAIN, training_source, src_lang, dst_lang)
    dataset = pd.read_csv(dataset_path)

    # data cleaning
    dataset.replace([np.inf], 99999, inplace=True)
    dataset = dataset.drop(['qid_tmp', 'index_tmp'], axis=1)
    
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
    X_scaler = MinMaxScaler()
    X_scaler.fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_val = X_scaler.transform(X_val)
    # X_test = scaler.transform(X_test) # transforming at inference time
    y_scalar = MinMaxScaler()
    y_scalar.fit(y_train)
    y_train = y_scalar.transform(y_train)
    y_val = y_scalar.transform(y_val)
    y_test = y_scalar.transform(y_test)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    # Add to device
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_scaler

# %% [markdown]
# ### Train Model

# %%
def get_ctq_scores(model, input_X, x_scalar, features, device):
    ctq_scores = []
    model.eval()
    with torch.no_grad():
        for i in range(len(input_X)):
            X_sample = input_X[i: i+1]
            X_sample = X_sample[features]
            X_sample = x_scalar.transform(X_sample)
            X_sample = torch.tensor(X_sample, dtype=torch.float32)
            X_sample = X_sample.to(device)
            y_pred = model(X_sample)
            ctq_scores.append(round(y_pred[0].item(), 4))

    return ctq_scores

# %%
def train_model(model, batch_size, n_epochs, optimizer, loss_fn, X_train, y_train, X_val, y_val, X_test, y_test, x_scalar, features, device, log_to_wandb=True):
    batch_start = torch.arange(0, len(X_train), batch_size)
    model.to(device)

    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=log_to_wandb) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
                
        # evaluate accuracy at end of each epoch
        model.eval()
        y_train_pred = model(X_train)
        train_mse = loss_fn(y_train_pred, y_train)
        train_mse = float(train_mse)
        
        y_val_pred = model(X_val)
        val_mse = loss_fn(y_val_pred, y_val)
        val_mse = float(val_mse)
        history.append(val_mse)
        if val_mse < best_mse:
            best_mse = val_mse
            best_weights = copy.deepcopy(model.state_dict())
            
        if log_to_wandb:
           wandb.log({"train_loss": train_mse, "val_loss": val_mse, "epoch": epoch, "best_mse": best_mse})


    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    # evaluate test accuracy at end of run using best weights
    y_test_pred = get_ctq_scores(model, X_test, x_scalar, features, device)
    test_mse = mean_squared_error(y_test, y_test_pred, squared=True)

    if log_to_wandb:
        wandb.log({"actual_test_loss": test_mse})
    print('Actual test loss: {}'.format(test_mse))    
    print("Val MSE: %.5f" % best_mse)

    # plt.plot(history)
    # plt.show()
    # plt.savefig('plots/test.png')

# %% [markdown]
# ### Generate best model using tuned hyperparameters

# %%
def get_best_model(X_train, y_train, X_val, y_val, X_test, y_test, x_scalar, features, device, src_lang, dst_lang,
                   activation_func_name, batch_size, learning_rate, neurons_hidden_layer,
                   no_of_hidden_layer, n_epochs, optimizer_func_name, weight_decay):
    
    input_size = len(features)
    output_size = 1

    set_seed()
    activation_func = get_activation_func(activation_func_name)
    model = NeuralNet(input_size, output_size, no_of_hidden_layer, neurons_hidden_layer, activation_func)
    model.to(device)
    optimizer = get_optimizer(model, optimizer_func_name, learning_rate, weight_decay)
    loss_fn = nn.MSELoss()

    train_model(model, batch_size, n_epochs, optimizer, loss_fn, X_train, y_train, X_val, y_val, X_test, y_test, x_scalar, features, device, log_to_wandb=False)
    
    # save the model
    make_dir(SAVED_MODELS)
    model_path = '{}/{}_{}'.format(SAVED_MODELS, src_lang, dst_lang)
    torch.save(model, model_path)
    return model

# %% [markdown]
# ### Generate CTQScorer ranking for test data

# %%
# load test dataset and compute ctq scores and rank accordingly
def generate_ctqscorer_ranking(training_source, testing_source, src_lang, dst_lang, x_scalar, features, device, model=None):
    
    if not model:
        # load the saved model
        model_path = '{}/{}_{}'.format(SAVED_MODELS, src_lang, dst_lang)
        model = torch.load(model_path)
    
    # load the test dataset
    test_data_path = '{}/{}_{}_{}_{}.csv'.format(DATASET_TEST, training_source, testing_source, src_lang, dst_lang)
    X_test_raw = pd.read_csv(test_data_path)
    X_test_raw.replace([np.inf], 99999, inplace=True)
        
    # generate CTQ scores
    ctq_scores = get_ctq_scores(model, X_test_raw, x_scalar, features, device)
    X_test_raw['ctq_score'] = ctq_scores
    X_test = X_test_raw.copy()
    
    # write prompt scores to file and clean the test outputs
    X_test['ctq_score'] = X_test['ctq_score'].apply(lambda x: round(x, 4))
    X_test = X_test.sort_values(by=['qid'])

    # sort the predicted scores 
    result = {}
    for i, row in X_test.iterrows():
        qid, index, pred_comet_score = row['qid'], row['index'], row['ctq_score']
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
    make_dir('{}/{}'.format(EXAMPLE_SELECTION_TEST_DATA, RANKINGS_REGRESSION))
    with open('{}/{}/recommendations_{}_{}_{}_{}.json'.format(EXAMPLE_SELECTION_TEST_DATA, RANKINGS_REGRESSION, training_source, testing_source, src_lang, dst_lang), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

# %% [markdown]
# ### Sweeps and sweep configuration

# %%
def get_sweep_config(training_source, src_lang, dst_lang):
    sweep_config = {
        "name" : "general_sweeps",
        'method': "bayes",
        'metric': {
            'name': 'test_loss',
            'goal': 'minimize'  
        },
        "parameters" : {
            "neurons_hidden_layer" : {
                "values" : [128, 256, 512]
            },
            "number_of_epochs" : {
                "values" : [20, 30, 40]
            },
            "activation" : {
                "values" : ["sigmoid" , "relu" , "tanh"]
            },
            "no_of_hidden_layer" : {
                "values" : [3, 4, 5]
            },
            "batch_size" :{
                "values" : [16, 32, 64]
            },
            "optimizer" : {
                "values" : ['adam', 'rmsprop', 'sgd']
            },
            "weight_decay" : {
                "values" : [0]
            },
            "learning_rate" : {
                "values" : [0.001, 0.005, 0.01]
            },
            "output_size": {
                "values" : [1]
            },
            "src_lang": {
                "values": [src_lang]
            },
            "dst_lang": {
                "values": [dst_lang]
            },
            "dataset_used": {
                "values": [training_source]
            }
        }
    }
    
    return sweep_config

# %%
def run_train_sweeps(X_train, y_train, X_val, y_val, X_test, y_test, x_scalar, features, device, config=None):
    
    with wandb.init(config=config) as run:
        config = wandb.config
        
        set_seed()
        sweep_name = 'hl_{}_bs_{}_ac_{}_{}'.format(config.no_of_hidden_layer, config.batch_size, config.activation, config.optimizer)
        run.name = sweep_name
        print(sweep_name)

        # Create custom network using the above config file
        activation_func = get_activation_func(config.activation)
        model = NeuralNet(len(features), config.output_size, config.no_of_hidden_layer, config.neurons_hidden_layer, activation_func)
        model.to(device)
        optimizer = get_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
        loss_fn = nn.MSELoss()
        
        train_model(model, config.batch_size, config.number_of_epochs, optimizer, loss_fn, X_train, y_train, X_val, y_val, X_test, y_test, x_scalar, features, device)

    wandb.finish()


def main():    
    # ### Main function
    init_logging('ctqscorer.log')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_src", help="training source to be used")
    parser.add_argument("--test_src", help="testing source to be used")
    parser.add_argument("--src_lang", help="source language")
    parser.add_argument("--dst_lang", help="destination language")
    parser.add_argument("--train", help="Is the model being trained?", action="store_true")
    
    # arguments in case of inference
    parser.add_argument("--activation", help="activation function name (relu/tanh/sigmoid)")
    parser.add_argument("--batch_size", type=int, help="batch size", default=64)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.01)
    parser.add_argument("--neurons_hidden_layer", type=int, help="number of neurons in each hidden layer", default=128)
    parser.add_argument("--no_of_hidden_layer", type=int, help="number of hidden layers", default=4)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=40)
    parser.add_argument("--optimizer", help="optimizer function name (adam,rmsprop,sgd)")
    parser.add_argument("--weight_decay", type=float, help="weight decay", default=0)
    args = parser.parse_args()
    
    # Inputs
    training_source = args.train_src
    testing_source = args.test_src
    src_lang = args.dst_lang
    dst_lang = args.src_lang

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

    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_val, y_val, X_test, y_test, x_scalar = data_preprocessing(training_source, src_lang, dst_lang, features, device)

    if args.train:
        # login to wandb, update your project, entity accordingly 
        wandb.login()
        sweep_config = get_sweep_config(training_source, src_lang, dst_lang)
        sweep_id = wandb.sweep(sweep_config, project="CTQScorer")

        # Code to train model
        try:
            wandb_train_func = functools.partial(run_train_sweeps, X_train, y_train, X_val, y_val, X_test, y_test, x_scalar, features, device)
            wandb.agent(sweep_id, function=wandb_train_func, count=40)
        except:
            pass

    else:
        # use the best tuned model and generate CTQScorer ranking
        ### update the below values based on the best hyperparameters in wandb
        activation_func_name = args.activation
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        neurons_hidden_layer = args.neurons_hidden_layer
        no_of_hidden_layer = args.no_of_hidden_layer
        n_epochs = args.epochs
        optimizer_func_name = args.optimizer
        weight_decay = args.weight_decay
        ### update the above values based on the best hyperparameters in wandb

        model = get_best_model(X_train, y_train, X_val, y_val, X_test, y_test, x_scalar, features, device, src_lang, dst_lang,
                            activation_func_name, batch_size, learning_rate, neurons_hidden_layer, no_of_hidden_layer, n_epochs, optimizer_func_name, weight_decay)

        # Use best model to predict CTQ scores for the test dataset
        generate_ctqscorer_ranking(training_source, testing_source, src_lang, dst_lang, x_scalar, features, device, model)


if __name__ == '__main__':
    main()

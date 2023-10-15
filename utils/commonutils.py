
import os
import logging
import json
from sentence_transformers import SentenceTransformer, util
import logging
import numpy as np
import torch
import random
import time
import string


lang_abbr_to_lang_code = {
    'asm_Beng' : 'as',
    'ben_Beng' : 'bn',
    'eng_Latn' : 'en',
    'guj_Gujr' : 'gu',
    'hin_Deva' : 'hi',
    'kan_Knda' : 'kn',
    'mal_Mlym' : 'ml',
    'mar_Deva' : 'mr',
    'npi_Deva' : 'ne',
    'ory_Orya' : 'or',
    'pan_Guru' : 'pa',
    'san_Deva' : 'sa',
    'snd_Arab' : 'sd',
    'tam_Taml' : 'ta',
    'tel_Telu' : 'te',
    'urd_Arab' : 'ur',
    'fra_Latn' : 'fr',
    'spa_Latn' : 'es',
    'deu_Latn' : 'de',
    'rus_Cyrl' : 'ru'
}


lang_abbr_to_lang = {
    'asm_Beng' : "Assamese",
    'ben_Beng' : "Bengali",
    'eng_Latn' : "English",
    'guj_Gujr' : "Gujarati",
    'hin_Deva' : "Hindi",
    'kan_Knda' : "Kannada",
    'mal_Mlym' : "Malayalam",
    'mar_Deva' : "Marathi",
    'ory_Orya' : "Oriya",
    'pan_Guru' : "Punjabi",
    "san_Deva" : "Sanskrit",
    'tam_Taml' : "Tamil",
    'tel_Telu' : "Telugu",
    'urd_Arab' : "Urdu",
    'npi_Deva' : "Nepali",
    'fra_Latn' : "French",
    'spa_Latn' : "Spanish",
    'deu_Latn' : "German",
    'rus_Cyrl' : "Russian"
}


def make_dir(path):
    """
    Make directories recursively till the path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def init_logging(filename):
    """
    Initiates logging.
    """
    make_dir('logs')
    logging.basicConfig(filename="logs/{}".format(filename), 
                        level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',)


def read_file(filepath):
    """
    Reads content of file and returns file.
    """
    content = ''
    with open(filepath, 'r') as f:
        content = f.read()
    return content


def load_samples(filepath):
    content = read_file(filepath)
    return content.splitlines()


def write_json_to_file(jsondata, file_path):
    with open(file_path, 'w') as f:
        json.dump(jsondata, f, indent=4, ensure_ascii=False)


def get_random_name():
    # generate random name for file to map the configuration
    random.seed(time.time())
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))


def append_config_to_file(filepath, msg):
    with open(filepath, 'a') as f:
        f.write('-' * 50 + '\n')
        f.write(msg)
        f.write('-' * 50 + '\n')


def get_embeddings(sentences):
    logging.info('generating embeddings...')
    model = SentenceTransformer("sentence-transformers/LaBSE")
    embeddings = model.encode(sentences)
    return embeddings


def get_cos_sim_for_embeddings(embeddings1, embeddings2):
    cos_sim_mat = util.cos_sim(embeddings1, embeddings2)
    cos_sim = []
    cos_sim.extend(torch.diag(cos_sim_mat, 0).tolist())
    return np.mean(cos_sim)


def get_cos_sim_for_sents(samples1, samples2):
    embeddings1 = get_embeddings(samples1)
    embeddings2 = get_embeddings(samples2)
    
    cos_sim_mat = util.cos_sim(embeddings1, embeddings2)
    cos_sim = []
    cos_sim.extend(torch.diag(cos_sim_mat, 0).tolist())
    return np.mean(cos_sim)
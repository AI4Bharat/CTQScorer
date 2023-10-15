from constants import *
from utils import lang_abbr_to_lang_code
import sys

def get_train_test_data(training_source, testing_source, src_lang, dst_lang, is_ranking_for_devset=False):
    src_lang_code = lang_abbr_to_lang_code.get(src_lang)
    dst_lang_code = lang_abbr_to_lang_code.get(dst_lang)
    folder_name = 'en-{}'.format(dst_lang_code) if dst_lang_code != 'en' else 'en-{}'.format(src_lang_code)            

    # Get train_src_path and train_dst_path
    if training_source == FLORES:
        train_src_path = 'dataset/train/{}.dev'.format(src_lang)
        train_dst_path = 'dataset/train/{}.dev'.format(dst_lang)
    elif training_source == SAMANANTAR:
        # en-hi is the folder name
        train_src_path = 'dataset/samanantar/{}/train.{}'.format(folder_name, src_lang_code)
        train_dst_path = 'dataset/samanantar/{}/train.{}'.format(folder_name, dst_lang_code)        
    elif training_source == EUROPARL:
        train_src_path = 'dataset/europarl/{}/train.{}'.format(folder_name, src_lang_code)
        train_dst_path = 'dataset/europarl/{}/train.{}'.format(folder_name, dst_lang_code)
    elif training_source == PARACRAWL:
        train_src_path = 'dataset/paracrawl/{}/train.{}'.format(folder_name, src_lang_code)
        train_dst_path = 'dataset/paracrawl/{}/train.{}'.format(folder_name, dst_lang_code)
    # New training_source needs to be included here
    else:
        print('Invalid Training source: {}'.format(training_source))
        sys.exit(1)

    # Get test_src_path and test_dst_path
    if testing_source == FLORES:
        if is_ranking_for_devset:
            test_src_path = 'dataset/train/{}.dev'.format(src_lang)   
            test_dst_path = 'dataset/train/{}.dev'.format(dst_lang)
        else:
            test_src_path = 'dataset/test/{}.devtest'.format(src_lang)
            test_dst_path = 'dataset/test/{}.devtest'.format(dst_lang)
    else:
        print('Invalid Testing source: {}'.format(testing_source))
        sys.exit(1)
        
    return train_src_path, train_dst_path, test_src_path, test_dst_path

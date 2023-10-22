python rankings_generate.py --train_src samanantar --test_src flores --src_lang hin_Deva --dst_lang eng_Latn --algorithm rankings_bm25_and_perplexity >> rankings_generate.log

# command to generate all algorithms using flores test set
# python rankings_generate.py --train_src samanantar --test_src flores --src_lang hin_Deva --dst_lang eng_Latn --algorithm all >> rankings_generate.log

# command to generate all algorithms using flores devtest set
# python rankings_generate.py --train_src samanantar --test_src flores --src_lang hin_Deva --dst_lang eng_Latn --algorithm all --devset >> rankings_generate.log

# command to generate all algorithms using flores test set using xglm model
# python rankings_generate.py --train_src samanantar --test_src flores --deu_Latn  --dst_lang eng_Latn --algorithm all --xglm >> rankings_generate.log

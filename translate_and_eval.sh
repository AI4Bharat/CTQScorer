# Add `--xglm` argument if translation is being performed for/to deu_Latn, fra_Latn, rus_Cyrl

$TRAIN_SRC="samanantar"
$TEST_SRC="flores"
$SRC_LANG="hin_Deva"
$DST_LANG="eng_Latn"
 
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_1 --algorithm random_selection >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_2 --algorithm rankings_bm25_and_reranking >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_3 --algorithm rankings_bm25 >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_4 --algorithm comet_qe_query_src_score >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_5 --algorithm comet_qe_query_dst_score >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_6 --algorithm comet_qe_src_dst_score >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_7 --algorithm rankings_bm25_and_chrf >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_8 --algorithm labse_score_query_src >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_9 --algorithm labse_score_query_dst >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_10 --algorithm labse_score_src_dst >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_11 --algorithm src_dst_ppl >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_12 --algorithm src_dst_query_ppl >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_13 --algorithm no_of_tokens_in_src_sent >> translate_and_eval.log
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_14 --algorithm no_of_tokens_in_dst_sent >> translate_and_eval.log
# experiment - ScAvg (3-feat)
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_15 --algorithm rankings_bm25_and_3_way >> translate_and_eval.log
# experiment - CTQ (3-feat)
python translate_and_eval.py --train_src $TRAIN_SRC --test_src $TEST_SRC --src_lang $SRC_LANG --dst_lang $DST_LANG --experiment exp_16 --algorithm rankings_bm25_and_3_way >> translate_and_eval.log

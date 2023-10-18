python compute_regression_scores.py --train samanantar --test flores --src ben_Beng --dst eng_Latn --outputs outputs/exp_reg3_bloom-7b1_Bengali_English_1_shots_pred_DTNNAUAR65.txt
python compute_regression_scores.py --train samanantar --test flores --src guj_Gujr --dst eng_Latn --outputs  outputs/exp_reg2_bloom-7b1_Gujarati_English_1_shots_pred_C56JKQYPTH.txt
python compute_regression_scores.py --train samanantar --test flores --src hin_Deva --dst eng_Latn --outputs outputs/exp_reg1_bloom-7b1_Hindi_English_1_shots_pred_KCH1P8V0AQ.txt

python compute_regression_scores.py --train samanantar --test flores --src eng_Latn --dst ben_Beng --outputs outputs/exp_reg6_bloom-7b1_English_Bengali_1_shots_pred_5BQD47SVNP.txt
python compute_regression_scores.py --train samanantar --test flores --src eng_Latn --dst guj_Gujr --outputs outputs/exp_reg5_bloom-7b1_English_Gujarati_1_shots_pred_36N37141HZ.txt
python compute_regression_scores.py --train samanantar --test flores --src eng_Latn --dst hin_Deva --outputs outputs/exp_reg4_bloom-7b1_English_Hindi_1_shots_pred_C31UL87ZTN.txt

python compute_regression_scores.py --train europarl --test flores --src fra_Latn --dst eng_Latn --outputs outputs/exp_reg11_xglm-7.5B_French_English_1_shots_pred_9L70XD5O18.txt
python compute_regression_scores.py --train europarl --test flores --src deu_Latn --dst eng_Latn --outputs outputs/exp_reg7_xglm-7.5B_German_English_1_shots_pred_7S4XADCR5N.txt
python compute_regression_scores.py --train paracrawl --test flores --src rus_Cyrl --dst eng_Latn --outputs outputs/exp_reg9_xglm-7.5B_Russian_English_1_shots_pred_CCKSAGSZ0W.txt

python compute_regression_scores.py --train europarl --test flores --src eng_Latn --dst fra_Latn --outputs outputs/exp_reg12_xglm-7.5B_English_French_1_shots_pred_HPJ8NDVS44.txt
python compute_regression_scores.py --train europarl --test flores --src eng_Latn --dst deu_Latn --outputs outputs/exp_reg8_xglm-7.5B_English_German_1_shots_pred_5EIO0EI5HB.txt
python compute_regression_scores.py --train paracrawl --test flores --src eng_Latn --dst rus_Cyrl --outputs outputs/exp_reg10_xglm-7.5B_English_Russian_1_shots_pred_DVUDKIFWG2.txt

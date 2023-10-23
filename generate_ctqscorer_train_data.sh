python generate_ctqscorer_train_data.py --train samanantar --test flores --src ben_Beng --dst eng_Latn
python generate_ctqscorer_train_data.py --train samanantar --test flores --src guj_Gujr --dst eng_Latn
python generate_ctqscorer_train_data.py --train samanantar --test flores --src hin_Deva --dst eng_Latn

python generate_ctqscorer_train_data.py --train samanantar --test flores --src eng_Latn --dst ben_Beng
python generate_ctqscorer_train_data.py --train samanantar --test flores --src eng_Latn --dst guj_Gujr
python generate_ctqscorer_train_data.py --train samanantar --test flores --src eng_Latn --dst hin_Deva

python generate_ctqscorer_train_data.py --train europarl --test flores --src fra_Latn --dst eng_Latn --xglm
python generate_ctqscorer_train_data.py --train europarl --test flores --src deu_Latn --dst eng_Latn --xglm
python generate_ctqscorer_train_data.py --train paracrawl --test flores --src rus_Cyrl --dst eng_Latn --xglm

python generate_ctqscorer_train_data.py --train europarl --test flores --src eng_Latn --dst fra_Latn --xglm
python generate_ctqscorer_train_data.py --train europarl --test flores --src eng_Latn --dst deu_Latn --xglm
python generate_ctqscorer_train_data.py --train paracrawl --test flores --src eng_Latn --dst rus_Cyrl --xglm

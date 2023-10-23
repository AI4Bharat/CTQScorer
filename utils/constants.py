IN22_OTHER_SOURCES = 'in22_other_sources'
SAMANANTAR = 'samanantar'
FLORES = 'flores'
WMT = 'wmt'
OPUS100 = 'opus100'
OPUSBOOKS = 'opusbooks'
EUROPARL = 'europarl'
PARACRAWL = 'paracrawl'

ASM_BENG = 'asm_Beng'
BEN_BENG = 'ben_Beng'
ENG_LATN = 'eng_Latn'
GUJ_GUJR = 'guj_Gujr'
HIN_DEVA = 'hin_Deva'
KAN_KNDA = 'kan_Knda'
MAL_MLYM = 'mal_Mlym'
MAR_DEVA = 'mar_Deva'
ORY_ORYA = 'ory_Orya'
PAN_GURU = 'pan_Guru'
SAN_DEVA = 'san_Deva'
TAM_TAML = 'tam_Taml'
TEL_TELU = 'tel_Telu'
URD_ARAB = 'urd_Arab'
NPI_DEVA = 'npi_Deva'
FRA_LATN = 'fra_Latn'
SPA_LATN = 'spa_Latn' 
DEU_LATN = 'deu_Latn'
RUS_CYRL = 'rus_Cyrl'

RANDOM_SELECTION = 'random_selection'
RANKINGS_BM25 = 'rankings_bm25'
RANKINGS_BM25_AND_RERANKING = 'rankings_bm25_and_reranking'
RANKINGS_COMET_QA = 'rankings_comet_qa'
RANKINGS_BM25_AND_LABSE = 'rankings_bm25_and_labse' # This same as query_src
RANKINGS_BM25_AND_LABSE_QUERY_DST = 'rankings_bm25_and_labse_query_dst'
RANKINGS_BM25_AND_LABSE_SRC_DST = 'rankings_bm25_and_labse_src_dst'
RANKINGS_BM25_AND_CHRF = 'rankings_bm25_and_chrf'
RANKINGS_BM25_AND_3_WAY = 'rankings_bm25_and_3_way'
RANKINGS_BM25_AND_3_WAY_ORACLE = 'rankings_bm25_and_3_way_oracle'
RANKINGS_BM25_AND_PERPLEXITY = 'rankings_bm25_and_perplexity'
SRC_PPL = 'src_ppl'
DST_PPL = 'dst_ppl'
RANKINGS_NO_OF_TOKENS = 'rankings_no_of_tokens'
RANKINGS_REGRESSION = 'rankings_regression'
RANKINGS_BM25_AND_DIVERSITY = 'rankings_bm25_and_diversity'
RANKINGS_3WAY_REGRESSION = 'rankings_3way_regression'

BLOOM_3B = "bigscience/bloom-3b"
BLOOM_7B = "bigscience/bloom-7b1"
XGLM_7B = "facebook/xglm-7.5B"

QUERY_SRC = 'query_src'
QUERY_DST = 'query_dst'
SRC_DST = 'src_dst'
ALL = 'all'

RANKINGS_CUSTOM = 'rankings_custom'
RANKINGS_LINEAR_REGRESSION = 'rankings_linear_regression'
ALL_FEATURES = 'all_features'
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
RANKINGS_BM25_REGRESSION = 'rankings_bm25_regression'
COMET_QE_20_REGRESSION = 'comet_qe_20_regression' 

BATCH_SIZE = 8


# constants for CTQScorer
COMET_SCORE = 'comet_score'
BLEU_SCORE = 'bleu_score'
SAVED_MODELS = 'saved_models'
DATASET_TRAIN = 'dataset_train'
DATASET_TEST = 'dataset_test'

training_source = EUROPARL
testing_source = FLORES
src_lang = FRA_LATN
dst_lang = ENG_LATN
approach = 'comet_qe_20_regression'

EXAMPLE_SELECTION_TEST_DATA = 'example_selection_test_data'
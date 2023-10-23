import json
import logging
from indicnlp.tokenize import indic_tokenize  
from collections import Counter
import math
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from indicnlp.tokenize import sentence_tokenize, indic_tokenize  

# nltk.download('stopwords')
# TODO: get stopwords for Indic languages

indic_langs = ["asm_Beng", "ben_Beng", "guj_Gujr", "hin_Deva", "kan_Knda", "mal_Mlym", "mar_Deva",
"ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu", 'brx_Deva', 'doi_Deva', 'kas_Arab', 
'kas_Deva', 'gom_Deva', 'mai_Deva', 'mni_Beng', 'npi_Deva', 'san_Deva', 'sat_Beng',
'snd_Arab', 'urd_Arab']

"""
Get stopwords for a language
"""
def get_stopwords(langAbbr):
    if langAbbr == 'eng_Latn':
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set()
    return stop_words


"""
Tokenize words using indic to
"""
def tokenize_words(line, langAbbr='eng_Latn'):
    if langAbbr == 'eng_Latn':
        tokens = word_tokenize(line)
    elif langAbbr in indic_langs:
        tokens = indic_tokenize.trivial_tokenize(line)
    else:
        tokens = line.split()
    return tokens


"""
Parsers the training corpus into documents 
"""
class CorpusParser:

    def __init__(self, filename, langAbbr='eng_Latn'):
        self.filename = filename
        # self.corpus = dict()
        self.org_corpus = dict()
        self.langAbbr = langAbbr
        self.stop_words = get_stopwords(langAbbr=langAbbr)

    def parse(self, required_docs):
        required_docs = list(required_docs)
        required_docs.sort()

        with open(self.filename, 'r') as f:
            s = f.read()
                
        docid = 0
        blobs = s.splitlines()
        for required in required_docs:
            self.org_corpus[required] = blobs[required]
        # for x in blobs:
        #     if docid % 10000 == 0:
        #         logging.info(docid)
        #     # text = tokenize_words(x, self.langAbbr)
        #     # filtered_text = [w for w in text if not w.lower() in self.stop_words]
        #     # self.corpus[docid] = filtered_text
        #     if docid in required_docs:
        #         self.org_corpus[docid] = x

        #     docid += 1

    # def get_corpus(self):
    #     return self.corpus

    def get_org_corpus(self):
        return self.org_corpus


"""
Parsers the queries document into queries
"""
class QueryParser:

    def __init__(self, filename, langAbbr='eng_Latn'):
        self.filename = filename
        # self.queries = []
        self.org_queries = []
        self.langAbbr = langAbbr
        self.stop_words = get_stopwords(langAbbr=langAbbr)


    def parse(self):
        with open(self.filename, 'r') as f:
            lines = f.read()

        for x in lines.split('\n'):
            text = tokenize_words(x, self.langAbbr)
            # filtered_text = [w for w in text if not w.lower() in self.stop_words]
            # self.queries.append(filtered_text)
            self.org_queries.append(x)

    # def get_queries(self):
    #     return self.queries 

    def get_org_queries(self):
        return self.org_queries



def extract_word_ngrams_with_count(indic_string, ngrams=2):
    x = indic_tokenize.trivial_tokenize(indic_string)
    stopwords={"है", "।" , "?", ".", ","}
    filtered = [ele for ele in x if ele not in stopwords]
    ngrams_zip = []
    for i in range(0, ngrams):
        ngrams_zip.append(filtered[i::])

    
    ngramsCounter = Counter(zip(*ngrams_zip))
    ngramsCounter = dict(ngramsCounter)
    return ngramsCounter

def extract_word_ngrams_with_count_upto_n(indic_string, ngrams_upto):
    ngramsCounter = {}
    for ngrams in range(1, ngrams_upto + 1):
        ngramsCounter.update(extract_word_ngrams_with_count(indic_string, ngrams))
    
    return ngramsCounter

def get_ngram_overlap_score_and_matched_ngrams(source_ngrams, query_ngrams, ngrams_upto=2):
    matched = {key: min(source_ngrams[key], query_ngrams[key]) for key in source_ngrams if key in query_ngrams }

    power_value = 0
    for ngrams in range(1, ngrams_upto + 1):
        ngrams_matched = {key: matched[key] for key in matched if len(key) == ngrams }
        ngrams_from_the_source = {key: source_ngrams[key] for key in source_ngrams if len(key) == ngrams }
        number_of_matched_ngrams = sum(ngrams_matched.values())
        number_of_source_ngrams = sum(ngrams_from_the_source.values())
        if number_of_matched_ngrams == 0:
            continue
        recall_ngrams = number_of_matched_ngrams / number_of_source_ngrams
        power_value = power_value + ((1 / 4) * ngrams * math.log(recall_ngrams))
        
    if power_value == 0:
        return 0,  {}
    score = math.e ** (power_value)
    score = round(score, 2)
    return score, matched


def rerankingAlgorithm(single_query_source, doc_ids, documents, upto_ngrams, number_of_shots=4, threshold=0, lambdavalue=0.25):
    T = []
    S = extract_word_ngrams_with_count_upto_n(single_query_source, upto_ngrams)

    Q = []
    # after updating the structure had to change to below
    doc_ids = list(map(lambda x: x["index"], doc_ids))
    for doc_id in doc_ids:
        Q.append(extract_word_ngrams_with_count_upto_n(documents[doc_id], upto_ngrams))

    while len(T) < number_of_shots:
        obj = []
        for doc_ngrams, doc_id in zip(Q, doc_ids):
            if len(doc_ngrams) == 0:
                obj.append([0, {}, 0])
                continue
            score, matched = get_ngram_overlap_score_and_matched_ngrams(S, doc_ngrams)
            obj.append([score, matched, doc_id])

        max_score = 0
        max_score_ind = 0
        max_score_matched = {}
        best_doc_id = 0
        for ind, item in enumerate(obj):
            score, matched, doc_id = item
            if max_score < score:
                # print(matched)
                max_score = score
                max_score_ind = ind
                max_score_matched = matched
                best_doc_id = doc_id

        if max_score <= threshold:
            break

        T.append(best_doc_id)
        Q[max_score_ind] = {}
        
        # print(matched)
        for ngram in max_score_matched.keys():
            S[ngram] = S[ngram] * lambdavalue
    # print('after: ' + str(S))

    # sometimes there may not be any overlap, in that case, we just append the bm25 rankings
    if len(T) < number_of_shots:
        ind = 0
        while ind < len(doc_ids) and len(T) < number_of_shots:
            if doc_ids[ind] not in T:
                T.append(doc_ids[ind])
            ind += 1        

    return T


def get_reranking_ranking(train_src_path, train_dst_path, test_src_path, test_dst_path, src_lang, dst_lang, bm25_file_name, is_ranking_for_devset=False):
    logging.debug('parsing queries & corpus...')
    qp = QueryParser(filename=test_src_path, langAbbr=src_lang)
    qp.parse()

    json_data = ''
    with open(bm25_file_name, 'r') as f:
        json_data = json.load(f)

    queries = qp.get_org_queries()
    required_docs = set()
    for qid, query in enumerate(queries):
        if query.strip() == '':
            continue
        one_bm25 = json_data[str(qid)]
        doc_ids = list(map(lambda x: x["index"], one_bm25))
        required_docs.update(doc_ids)

    cp = CorpusParser(filename=train_src_path, langAbbr=src_lang)
    cp.parse(required_docs)
    cid_to_corpus = cp.get_org_corpus()
    logging.debug('parsed queries & corpus!')
    logging.info('number of required docs: {}'.format(len(required_docs)))
    logging.info('number of samples is: {}'.format(len(cid_to_corpus)))

    rerankings = {}
    logging.debug('reranking samples...')
    for qid, query in enumerate(queries):
        if query.strip() == '':
            continue
        logging.info('reranking for query: {}'.format(qid))
        doc_ids = rerankingAlgorithm(query, json_data[str(qid)], cid_to_corpus, upto_ngrams=3, number_of_shots=8, threshold=0, lambdavalue=0.25)
        rerankings[int(qid)] = doc_ids
    logging.debug('reranked samples!')

    return rerankings

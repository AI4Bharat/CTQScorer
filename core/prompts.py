import random
from .model_parameters import model_parameters


# returns n-shot example for the given source and target languages.
# we can pass recommendations for an input sample obtained from BM25 & reranking algorithm.
def get_n_shots(mp: model_parameters, src_samples, dst_samples, n_shots, src_lang, dst_lang, recommendations=[]):

    # start_time = time.time()
    # sometimes the recommendations from BM25 is less than n-shots
    # then we randomly choose samples from the dev dataset
    random.seed(mp.seed)
    random_numbers = recommendations

    # Don't add sentences larger than 120 words
    THRESHOLD = 120
    for random_number in random_numbers:
        sent = src_samples[random_number].strip('"').split()
        if len(sent) > THRESHOLD:
            random_numbers.remove(random_number)

    while(len(random_numbers) < n_shots):
        x = random.randint(0,len(src_samples) - 1)
        sent = src_samples[x].strip('"').split()
        if x in random_numbers or len(sent) > THRESHOLD:
            continue
        random_numbers.append(x)

    content = ''

    count = 0
    i = 0
    while count < n_shots and i < len(random_numbers):
        sent = src_samples[random_numbers[i]].strip('"').split()
        src_sample = src_samples[random_numbers[i]].strip('"')
        dst_sample = dst_samples[random_numbers[i]].strip('"')

        if len(sent) < THRESHOLD:
            count += 1
            if n_shots == 1:
                content = content + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, src_sample, dst_lang, dst_sample)
            else:
                content = content + """{} Sentence: "{}"
{} Sentence: "{}"
###
""".format(src_lang, src_sample, dst_lang, dst_sample)
        i += 1

    return content


# This function concatenates the n-shots and the given input sample
def construct_prompt(shots, input_sample, src_lang, dst_lang, n_shots=0):
    if n_shots == 1:
        return shots + """{} Sentence: "{}"
{} Sentence: """.format(src_lang, input_sample.strip('"'), dst_lang)
    return shots + """{} Sentence: "{}"
{} Sentence: """.format(src_lang, input_sample.strip('"'), dst_lang)


# This function generates zero shot example
def construct_zero_shot(input_sample, src_lang, dst_lang):
    return """Translate {} Sentence: "{}" to {} Sentence: """.format(src_lang, input_sample.strip('"'), dst_lang)
import re
from collections import Counter

# Preprocessing functions: Eliminates repetitive examples and similar sents from prompts
def has_similar_no_of_tokens(existing, current):
    matched = {key: min(existing[key], current[key]) for key in current if key in existing }
    total_no_tokens_matched = sum(matched.values())
    total_no_tokens_in_current = sum(current.values())
    # For some reason we get empty strings some times
    if total_no_tokens_in_current == 0:
        return True
    similarity = total_no_tokens_matched / total_no_tokens_in_current
    return True if similarity >= 0.8 else False

def check_similar_sent_exists_in_group(existing_group, current):
    for existing in existing_group:
        if has_similar_no_of_tokens(existing, current) or has_similar_no_of_tokens(current, existing):
            return True
    return False

def handle_repetitive_examples(src_train_samples, dst_train_samples, recommendations):
    filtered_indexes = []
    src_group = []
    dst_group = []
    
    for index in recommendations:
        src_sent = re.sub('[?,.!।]+', '', src_train_samples[index].lower())
        dst_sent = re.sub('[?,.!।]+', '', dst_train_samples[index].lower())

        src_tokens_counter = dict(Counter(src_sent.split()))
        dst_tokens_counter = dict(Counter(dst_sent.split()))
        
        if check_similar_sent_exists_in_group(src_group, src_tokens_counter) \
        or check_similar_sent_exists_in_group(dst_group, dst_tokens_counter):
            continue
        else:
            src_group.append(src_tokens_counter)
            dst_group.append(dst_tokens_counter)
            filtered_indexes.append(index)
    
    return filtered_indexes
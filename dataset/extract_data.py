import os

def extract_data_from_europarl(src_lang, dst_lang):
    print('extracting europarl data for language: {}-{}'.format(src_lang, dst_lang))
    filename = 'europarl-v10.{}-{}.tsv'.format(src_lang, dst_lang)
    content = ''
    with open(filename, 'r') as f:
        content = f.read()
    content = content.splitlines()
    folder = 'europarl/{}-{}'.format(dst_lang, src_lang)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open('{}/train.{}'.format(folder, src_lang), 'w') as f:
        f.write('')
        
    with open('{}/train.{}'.format(folder, dst_lang), 'w') as f:
        f.write('')
    
    for line in content:
        line = line.split('\t')
        # print(line)
        src_line, dst_line = line[0], line[1]
        
        with open('{}/train.{}'.format(folder, src_lang), 'a') as f:
            f.write('{}\n'.format(src_line))
            
        with open('{}/train.{}'.format(folder, dst_lang), 'a') as f:
            f.write('{}\n'.format(dst_line)) 


def extract_data_from_paracrawl(src_lang, dst_lang):
    print('extracting paracrawl data for language: {}-{}'.format(src_lang, dst_lang))
    filename = '{}-{}.txt'.format(src_lang, dst_lang)
    content = ''
    with open(filename, 'r') as f:
        content = f.read()
    content = content.splitlines()
    folder = 'paracrawl/{}-{}'.format(src_lang, dst_lang)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open('{}/train.{}'.format(folder, src_lang), 'w') as f:
        f.write('')
        
    with open('{}/train.{}'.format(folder, dst_lang), 'w') as f:
        f.write('')        
    
    for line in content:
        line = line.split('\t')
        # print(line)
        src_line, dst_line = line[0], line[1]
        
        with open('{}/train.{}'.format(folder, src_lang), 'a') as f:
            f.write('{}\n'.format(src_line))
            
        with open('{}/train.{}'.format(folder, dst_lang), 'a') as f:
            f.write('{}\n'.format(dst_line))
             

# Extract Europarl datasets
extract_data_from_europarl(src_lang='fr', dst_lang='en')
extract_data_from_europarl(src_lang='de', dst_lang='en')

# Extract Paracrawl dataset
extract_data_from_paracrawl(src_lang='en', dst_lang='ru')
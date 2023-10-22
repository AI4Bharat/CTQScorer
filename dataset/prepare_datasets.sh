# Download datasets - Samanantar, Europarl, Paracrawl
wget -O samanantar.zip 'https://ai4b-my.sharepoint.com/:u:/g/personal/sumanthdoddapaneni_ai4bharat_org/EXhX84sbTQhLrsURCU9DlUwBVyJ10cYK9bQQe1SMljf_yA?e=q7GJpb&download=1'
wget https://www.statmt.org/europarl/v10/training/europarl-v10.fr-en.tsv.gz
wget https://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz
wget https://s3.amazonaws.com/web-language-models/paracrawl/bonus/en-ru.txt.gz

# extract Samanantar dataset
unzip samanantar.zip
mv v2 samanantar

# extract Europarl and Paracrawl datasets
mkdir europarl
mkdir paracrawl
gzip -d europarl-v10.fr-en.tsv.gz
gzip -d europarl-v10.de-en.tsv.gz
gzip -d en-ru.txt.gz
python extract_data.py

# Clean up redundant files
rm europarl-v10.de-en.tsv || true
rm europarl-v10.fr-en.tsv || true
rm en-ru.txt || true
rm -r samanantar/en-te/en-* || true
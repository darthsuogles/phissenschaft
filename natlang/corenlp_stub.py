''' CoreNLP Service Stub
'''
# wget --post-data 'the quick brown fox jumped over the lazy dog' 'localhost:9000/?properties={"annotators": "tokenize,ssplit,pos", "outputFormat": "json"}' -O -

import requests

endpoint = 'http://localhost:9000'

props = {
    "annotators": ','.join(["tokenize", "ssplit", "pos"]),
    "outputFormat": "json"
}
url = requests.get(endpoint, params=props).url
req = requests.post(url, 
                    data='the quick brown fox jumped over the lazy dog')
# The result is a JSON with all the requested features
parsed = req.json()

nouns = [tok for tok in parsed['sentences'][0]['tokens'] if 'NN' == tok['pos']]

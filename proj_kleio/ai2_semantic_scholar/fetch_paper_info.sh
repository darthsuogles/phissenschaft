#!/bin/bash

user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/601.4.4 (KHTML, like Gecko) Version/9.0.3 Safari/601.4.4"

article_title="A Bayesian Model for Supervised Clustering with the Dirichlet Process Prior"

cat <<EOF > params.json
{"autoEnableFilters":true,
 "queryString":"${article_title}",
 "page":1,
 "pageSize":10,
 "sort":"relevance",
 "authors":[],
 "classifications":[],
 "coAuthors":[],
 "dataSets":[],
 "keyPhrases":[],
 "venues":[]}
EOF

curl -H "Content-Type: application/json" -X POST \
     --data @params.json \
     -A "${user_agent}" https://www.semanticscholar.org/api/1/search

# Then search the actual data with the paper's id and slug
#curl -L "https://www.semanticscholar.org/api/1/paper/a80c3c251bd9253dd70b9370bc05259468475a03?slug=Data-Clustering-50-Years-Beyond-K-means-Jain&citingPapersSort=is-influential&citedPapersSort=is-influential&citationsPageSize=10"

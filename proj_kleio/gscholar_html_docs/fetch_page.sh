#!/bin/bash

query="Expander graphs and their applications"

query_str=`echo $query | perl -ne 'chomp($line = $_); $line =~ s/\s+/\+/g; print $line;'`

#header='Mozilla/5.0 (X11; U; FreeBSD i386; en-US; rv:1.9.2.9) Gecko/20100913 Firefox/3.6.9'
#header="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.48 Safari/537.36"
header="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.73.11 (KHTML, like Gecko) Version/7.0.1 Safari/537.73.11"

# curl -v -A "$header" \
#     "http://scholar.google.com/scholar?hl=en&q=Kronecker+delta&btnG=&as_sdt=1%2C21&as_sdtp=" \
#     > kronecker_delta.html

curl -v -A "$header" \
    "http://scholar.google.com/scholar?hl=en&q=${query_str}&btnG=&as_sdt=1%2C21&as_sdtp=" \
    > ${query_str}.html

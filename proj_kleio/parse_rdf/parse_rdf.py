#!/usr/bin/env python

###
# Parsing a gzipped RDF file
#
###

import gzip

f = gzip.open('test_freebase.rdf.gz')
file_content = f.read()
f.close()

print file_content

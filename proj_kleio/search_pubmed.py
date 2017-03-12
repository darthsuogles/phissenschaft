#!/usr/bin/env python
##########################################
# Fetch contents from pubmed
# http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc109
# http://biopython.org/DIST/docs/tutorial/Tutorial.html#sec118
# http://www.ncbi.nlm.nih.gov/books/NBK25500/

import optparse
import sys
import re
#import urllib
#import urllib2
#from BeautifulSoup import BeautifulSoup

from Bio import Entrez
from research_article import Article

## Setup the account for Entrez
Entrez.email = "phi@cs.umd.edu"
Entrez.tool = "biopython"

if __name__ == "__main__":

    usage_str = """
        python search_pubmed.py [options] <query string> \n
        A command-line interface to pubmed search
        Text output will show on the terminal screen.
        """

    fmt = optparse.IndentedHelpFormatter(max_help_position=50,
                                         width=100)
    parser = optparse.OptionParser(usage=usage_str, formatter=fmt)
    parser.add_option('-a', '--author',
                      help='Author name')
    parser.add_option('--csv', action='store_true',
                      help='Print article data in CSV format (separator is "|")')
    parser.add_option('--csv-header', action='store_true',
                      help='Like --csv, but print header line with column names')
    parser.add_option('--txt', action='store_true',
                      help='Print article data in text format')
    parser.add_option('-c', '--count', type='int',
                      help='Maximum number of results')
    parser.add_option('-y', '--year', type='int',
                      help = 'Search for articles published since year xxxx')
    parser.set_defaults(count=7, author='')
    options, args = parser.parse_args()

    if len(args) < 1:
        print args
        print usage_str
        sys.exit(0)

    key_words = ' '.join(args)
    print "Key words: '%s'\n" % key_words
    # try:
    #     handle = Entrez.egquery(term = key_words)
    # except Exception as e:
    #     #pass
    #     print e.geturl()

    # record = Entrez.read(handle)
    # for row in record["eGQueryResult"]:
    #     if row["Status"] == "Ok":
    #         print row
    #     if row["DbName"] == "pubmed":
    #         print '# results from pubmed:', row["Count"]

    article_db = 'pubmed'
    article_link_name = article_db + "_" + article_db
    handle = Entrez.einfo()
    record = Entrez.read(handle)
    for k,v in record.iteritems():
        #print k, v
        if k == "Dblist":            
            assert article_db in record, "db(pubmed) is not available"            

    handle = Entrez.esearch(db = article_db, retmax = options.count, term = key_words)
    record = Entrez.read(handle)
    handle.close()

    # # Finding related articles of the one selected
    # for article_id in record["IdList"]:
    #     elhd = Entrez.elink(dbfrom = article_db, id = article_id, linkname = article_link_name)
    #     rec = Entrez.read(elhd)
    #     elhd.close()
    #     for elem in rec:
    #         print elem
    #         print "----------------------------\n"
    
    # for article_id in record["IdList"]:
    #     esum_hd = Entrez.esummary(db = article_db, id = article_id)
    #     rec = Entrez.read(esum_hd)
    #     esum_hd.close()
    #     rec = rec[0]
    #     print "Id:", rec['Id']
    #     print "Title:", rec['Title']
    #     print "Authors:", 
    #     for author in rec['AuthorList']:
    #         print author + ',',
    #     print
    #     print "Publication:", rec['Source'], rec['PubDate']
    #     print "Reference count:", rec['PmcRefCount']        
    #     if 'doi' in rec: 
    #         print "doi:", rec['doi']
    #     print "----------------------------\n"
    
    # Print in a org-mode friendly manner
    for article_id in record["IdList"]:
        esum_hd = Entrez.esummary(db = article_db, id = article_id)
        rec = Entrez.read(esum_hd)
        esum_hd.close()
        rec = rec[0]
        print "***", rec['Title']
        print '    :PROPERTIES:'
        print '    :CUSTOM_ID:', rec['Id']
        print "    :AUTHORS:", 
        for author in rec['AuthorList']:
            print author + ',',
        print
        print "    :PUBLICATION:", rec['Source'], rec['PubDate']
        print "    :REFCOUNT:", rec['PmcRefCount']        
        if 'doi' in rec: 
            print '    :DOI:', rec['doi']
        print '    :END:'
        print "----------------------------\n"

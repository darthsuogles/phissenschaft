#!/usr/bin/env python

##
# Retrieve a DOI given 
# 1. the first author
# 2. the title of the paper
#
# Now CrossRef provides a free API
# Ref: help.crossref.org/#using_http_to_post
##

## 
# Tips: 
#  C-c C-c to send the buffer
#  C-h m to view a description of current mode
## 

import optparse
import re
import urllib, urllib2
from collections import OrderedDict # to keep urlencode in order
from bs4 import BeautifulSoup

'''
curl -v -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36" --data "queryType=author-title&auth2=Menon+V&atitle2=Developmental+pathways+to+functional+brain+networks%3A+emerging+principles&multi_hit=true&article_title_search=Search" http://www.crossref.org/guestquery/
'''

user_agent_str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36"
#form_input = "queryType=author-title&auth2=Menon+V&atitle2=Developmental+pathways+to+functional+brain+networks%3A+emerging+principles&multi_hit=true&article_title_search=Search"
form_url = "http://www.crossref.org/guestquery/"
pid = "phi@cs.umd.edu"
api_url = "http://www.crossref.org/openurl?pid=" + pid + "&"

def form_query(authors, title):
    """
    issn
    title
    aulast: last name (preferably of first author)
    volume
    issue
    spage: first page
    date: publication year (YYYY)
    stitle: short title

    redirect: set to false to return the DOI in XML format
    multihit: set to true to return potentially more than one DOIs
    format: set to "unixref" to return metadata in UNIXREF format
    """
    data = OrderedDict([
        ("queryType", "author-title"),
        ("auth2", ','.join(authors)),
        ("atitle2", title),
        ("multi_hit", "true"),
        ("article_title_search", "Search")
    ])
    
    return urllib.urlencode(data)


def retrieve_doi(authors, title, is_regex = False):
    assert type(authors) == list

    # Fetch the html content 
    # Ref: http://docs.python.org/2/howto/urllib2.html 
    req = urllib2.Request(form_url,
                          form_query(authors, title),
                          headers = {'User-Agent': user_agent_str})
    print(req.get_full_url())
    print(req.get_data())
    response = urllib2.urlopen(req) 
    html_content = response.read()
    # fout = open('doi_sample.html', 'wb')
    # fout.write(html_content)
    # fout.close()
    
    ## Parse the content
    ## Ref: http://www.crummy.com/software/BeautifulSoup/ 
    #soup = BeautifulSoup(html_content) 
    ##print(soup.prettify())
    ## http://dx.doi.org/10.1016%2Fj.tics.2013.09.015
    if is_regex:
        return find_doi_string_regex(html_content)
    
    return find_doi_string(html_content)


def find_doi_string_regex(doi_html_raw):
    """
    Find the DOI string from an html file using regular expression
    (Deprecated)
    
    ref: http://docs.python.org/2/howto/regex.html#regex-howto 
    Warning: 
       This does not always return a valid match
       The result depend on the external site which might change     
    """
    repat = re.compile('http://dx\.doi\.org/[^<>\s]+') 
    m = repat.search(doi_html_raw) 
    #if m != None: 
    #    print m.group(0)
    doi = None if m is None else m.group(0)
    return doi


def find_doi_string(doi_html_raw):
    """
    Find the DOI string from an html file
    """
    html_content = BeautifulSoup( doi_html_raw )
    tables = html_content.findAll('table')
    doi_string = ''
    for tbl in tables:
        rows = tbl.findAll('tr')
        for tr in rows:
            cols = tr.findAll('td')
            for td in cols:
                try:
                    text = ''.join(td.find(text = True))
                    text = text.strip()
                    #print(text)
                    if text[:18] == "http://dx.doi.org/":
                        doi_string = text
                        return doi_string
                except TypeError:
                    text = None
                    pass

    return None


def cmdline():
    usage_str = """ 
    python search_arxiv.py [options] <query string> 
    
      A command-line interface to pubmed search.
      Text output will show on the terminal screen. """

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
                      help = 'Search for articles published since year YEAR')
    parser.set_defaults(count=7, author='')
    options, args = parser.parse_args()


if __name__ == "__main__":
    
    #cmdline()

    print user_agent_str

    #form_input = form_query(['Menon V'], 'Developmental pathways to functional brain networks')
    #print form_input
    print retrieve_doi(['Menon V'], 'Developmental pathways to functional brain networks')
    

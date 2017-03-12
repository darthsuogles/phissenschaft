#!/usr/bin/env python

import optparse
import urllib
import urllib2
import feedparser
import time

"""
http://export.arxiv.org/api/query?search_query=<prefix>:<key_word>

prefix	explanation
ti	 Title
au	 Author
abs	 Abstract
co	 Comment
jr	 Journal Reference
cat	 Subject Category
rn	 Report Number
id	 Id (use id_list instead)
all	 All of the above

Here are some useful techniques to form more complex queries

search query boolean operators
AND
OR
ANDNOT

search query grouping operators
symbol	encoding	explanation
( )	 %28 %29	 Used to group Boolean expressions for Boolean operator precedence.
\" \"	 %22 %22	 Used to group multiple words into phrases to search a particular field.
space	 +	         Used to extend a search_query to include multiple fields.

"""

UA = 'Mozilla/5.0 (X11; U; FreeBSD i386; en-US; rv:1.9.2.9) Gecko/20100913 Firefox/3.6.9'

def constr_arxiv_query(keywords,
                       excluded_keywords="",
                       start = 0, max_results = 10):
    url_pref = 'http://export.arxiv.org/api/query?search_query=all:'
    #url_suff = '&start=0&max_results=10'
    url_suff = urllib.urlencode([('start', start), ('max_results', max_results)])
    neu_qstr = ''
    lparen_cnt = 0
    for elem in keywords.split():
        if not elem:
            continue

        if not neu_qstr:
            neu_qstr += urllib.quote(elem)
            #neu_qstr += pref + elem + suff
        else:
            neu_qstr += '+AND+' + urllib.quote(elem)
            #neu_qstr += '+AND+' + pref + elem + suff

    if excluded_keywords:
        for elem in excluded_keywords.split():
            if not elem:
                continue

            assert neu_qstr, 'must include some key words in the query'

            neu_qstr += '+ANDNOT+' + urllib.quote(elem)

    return url_pref + neu_qstr + '&' + url_suff
        

# Parse the returned search result
def parse_arxiv_html(html):
    # Opensearch metadata such as totalResults, startIndex, 
    # and itemsPerPage live in the opensearch namespase.
    # Some entry metadata lives in the arXiv namespace.
    # This is a hack to expose both of these namespaces in
    # feedparser v4.1
    feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
    feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

    feed = feedparser.parse(html)

    # print out feed information 
    print 'Feed title: %s' % feed.feed.title
    print 'Feed last updated: %s' % feed.feed.updated

    # print opensearch metadata
    print 'totalResults for this query: %s' % feed.feed.opensearch_totalresults
    print 'itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage
    print 'startIndex for this query: %s'   % feed.feed.opensearch_startindex


    for ent in feed.entries:
        print 'e-print metadata'
        print 'arxiv-id: %s' % ent.id.split('/abs/')[-1]
        print 'Published: %s' % ent.published
        print 'Title:  %s' % ent.title
        
        print 'Tags:',
        for tag in feed.entries[0].tags:
            print tag.term,
        print

        author = ent.author
        # grab the affiliation in <arxiv:affiliation> if present
        # - this will only grab the first affiliation encountered
        #   (the first affiliation for the first author)
        # Please email the list with a way to get all of this information!
        try:
            author += ' (%s)' % ent.arxiv_affiliation
        except AttributeError:
            pass

        print 'last Author: ', author

        # feedparser v5.0.1 correctly handles multiple authors, print them all
        try:
            print 'Authors:  %s' % ', '.join(author.name for author in ent.authors)
        except AttributeError:
            pass

        # Get the links to abs page and pdf
        for link in ent.links:
            if link.rel == 'alternate':
                print 'abs page link:', link.href
            elif link.title == 'pdf':
                print 'pdf link:', link.href

        # The journal reference, comments and primary_category sections live under 
        # the arxiv namespace        
        try:
            journal_ref = ent.arxiv_journal_ref
        except AttributeError:
            journal_ref = 'No journal ref found'
        print 'Journal reference: %s' % journal_ref
            
        try:
            comment = ent.arxiv_comment
        except AttributeError:
            comment = 'No comment found'
        print 'Comments: %s' % comment

                    
        # The abstract is in the <summary> element
        print 'Abstract: %s' %  ent.summary
        print '\n\n'
    


if __name__ == "__main__":


    usage_str = """
        python search_arxiv.py [options] <query string> \n
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

    #print options, args

    paper_title = ' '.join(args)
    print paper_title
    #url = constr_arxiv_query('neuroimaging (machine "learning") optimization', 'EEG')
    url = constr_arxiv_query(paper_title, max_results = 10)
    req = urllib2.Request(url=url,
                          headers={'User-Agent': UA})
    hdl = urllib2.urlopen(req)
    html = hdl.read()
    parse_arxiv_html( html )

    # start = 0
    # results_per_iter = 5
    # total_results = 20
    # wait_time = 3

    # for pg in range(start, total_results, results_per_iter):

    #     print "Results %i - %i" % (pg, pg + results_per_iter)

    #     url = constr_arxiv_query('neuroimaging machine learning',
    #                              start = pg, max_results = results_per_iter)    
    #     req = urllib2.Request(url=url,
    #                           headers={'User-Agent': UA})

    #     html = urllib2.urlopen(req).read() 
    #     parse_arxiv_html( html )

    #     time.sleep(wait_time)

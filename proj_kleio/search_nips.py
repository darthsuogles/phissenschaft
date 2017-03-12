#!/usr/bin/env python
"""
Parse the papers in the NIPS conference collection

We parse images from 
This is an example of how to search text and images in PDF
"""

from HTMLParser import HTMLParser
from bs4 import BeautifulSoup
import sys

from research_article import Article

mandatory_fields = {0:'title', 1:'author', 2:'id'}
nips_year_zero = 1987
nips_db = [] # Store the whole database

# NIPS paper structure
class NIPS_Paper(Article):
    """
    A class representing articles from NIPS.
    This is a subset of the generic research Article
    """       
    def __init__(self):
        #self.attrs = {}
        #super(NIPS_Paper, self).__init__()
        Article.__init__(self)
        self.attrs['publication'] = 'Advances in Neural Information Processing Systems (NIPS)'                

    def gen_custom_id(self, meta_fields):
        return self.attrs['id']

# Create a subclass and override the handler methods
class NIPS_HTMLParser(HTMLParser):
    _tag_table_cnt = 0
    _is_processing = False
    _parse_index = 0
    curr_paper = None
    curr_url = None

    def handle_starttag(self, tag, attrs):
        global nips_db
        if tag == 'table':
            #print "Encountered a start tag:", tag
            self._tag_table_cnt += 1
            if self._tag_table_cnt > 1: 
                self._is_processing = True
            return

        if not self._is_processing:
            return
        
        if tag == 'a':
            href = attrs[0]
            if not href or href[0] != 'href':
                return

            if self.curr_paper:
                url = href[1]
                #if 'id' not in self.curr_paper.attrs.keys():
                if self.curr_paper['id'] == None:
                    path = url.split('://')
                    assert len(path) == 2, 'malformated url'
                    if path[0] not in ['http', 'https', 'ftp']:
                        print >>sys.stderr, 'Error: path %s is not valid' % url                
                        return                            
                    path = path[1].split('/')
                    assert len(path) > 1, 'malformated path to file'
                    if path[-1][:4].lower() != 'nips':
                        year = int(path[-2][4:]) + nips_year_zero
                        self.curr_paper['year'] = year
                        path = path[-2][:4].upper() + str(year) + '_' + path[-1]
                    else:
                        path = path[-1]
                    path = path.split('.')
                    assert len(path) > 1, 'malformated file name'
                    self.curr_paper['id'] = path[0]
                    # Add the year if we haven't captured it
                    if self.curr_paper['year'] == None:
                        curr_id = path[0]
                        assert curr_id[:4].lower() == 'nips'
                        assert len(curr_id) > 4
                        self.curr_paper['year'] = curr_id[4:curr_id.index('_')]
                        
                                        
                self.curr_url = url
            
            #print 'Found url:', href[1]

        # Begin parsing a new paper 
        elif tag == 'b': 
            # Finish processing the last paper, if any
            if self.curr_paper:
                if self.curr_paper.validate():
                    #print self.curr_paper.attrs
                    #print
                    #nips_db += [ self.curr_paper.attrs ]                    
                    nips_db += [ self.curr_paper ]
                #else:                    
                    #print >>sys.stderr, 'Warning: invalid paper entry'
                    #print >>sys.stderr, self.curr_paper.attrs
                self._parse_index = 0

            self.curr_paper = NIPS_Paper()
            #self.curr_url = None # reset url
            
    def handle_endtag(self, tag):
        if tag == 'table':
            #print "Encountered an end tag :", tag
            self._is_processing = False            

        if not self._is_processing:
            return        

    def handle_data(self, data):
        if not self._is_processing:
            return
        data = data.strip()
        if not data:
            return

        idx = self._parse_index

        if self.curr_url:
            data = data.strip('][')
            #self.curr_paper[data] = self.curr_url
            href = self.curr_url
            self.curr_paper[data] = href if isinstance(href, unicode) else href.decode('utf-8')
            self.curr_url = None

        elif idx < len(mandatory_fields):
            field = mandatory_fields[idx]            
            if field == 'author':
                author_list = data.split(', ')
                if author_list:
                    #self.curr_paper[field] = author_list
                    self.curr_paper[field] = [aut if isinstance(aut, unicode) else aut.decode('utf-8') 
                                              for aut in author_list]
            elif field == 'title':
                self.curr_paper[field] = data if isinstance(data, unicode) else data.decode('utf-8')

        self._parse_index += 1


# Find stuffs in the title
def query(nips_db, query_str):
    #results = [ ent for ent in nips_db if query_str in ent['title'].lower() ]
    results = [ art for art in nips_db if query_str in art['title'].lower() ]
    print 'Found %d papers containing: %s' % (len(results), query_str)
    for art in results:
        print '---------------------------------------'
        art.as_org_print()

#if __name__ == "__main__":
# instantiate the parser and fed it some HTML
parser = NIPS_HTMLParser()

for i in range(0, 25+1):
    fname = 'nips_papers/nips'+ str(i) +'.html'    
    fin = open(fname, 'r')
    html = fin.read()
    nips_papers = parser.feed(html)
    
print 'Total number of papers published in NIPS:', len(nips_db)
query(nips_db, 'forest')
    

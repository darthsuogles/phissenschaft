#!/usr/bin/env python
"""
Given a the references in the plain text format, 
parse the file into something of a bibtex format.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

import references as pdref


class Article(object):
    def __init__(self, ref_id, author_list, title, venue, year = -1): 
        # Check the types of input
        for v, t in [(ref_id, int), 
                     (author_list, list), 
                     (title, str), 
                     (venue, str),
                     (year, int)]:
            assert type(t) == type, "t must be a type"
            assert type(v) == t, "input argument must match type"
        
        self.ref_id = ref_id
        self.author_list = author_list
        self.title = title
        self.venue = venue
        self.year = year

    def __str__(self):
        return "ID:\t" + str(self.ref_id) + "\n" + \
               "YEAR:\t" + str(self.year) + "\n" + \
               "TITLE:\t" + self.title + "\n" + \
               "AUTHOR:\t" + ' | '.join(self.author_list)


class ArticleDB:
    """
    Store a list of reference articles and process their
    """
    def __init__(self, article_list):
        self.article_list = article_list
        
    def get_authors(self):
        author_dict = {}
        for article in self.article_list:
            for author in article.author_list:
                if author not in author_dict:
                    author_dict[author] = 1
                else:
                    author_dict[author] += 1
        #print sorted(author_dict.keys())
        #plt.hist(author_dict.values())
        #plt.show()

    def get_years(self):
        year_dict = {}
        for article in self.article_list:
            year = article.year
            if year not in year_dict:
                year_dict[year] = 1
            else:
                year_dict[year] += 1
        y0 = min(year_dict.keys())
        y1 = max(year_dict.keys())
        #print year_dict.values()
        #plt.scatter(year_dict.keys(), year_dict.values(), marker = '|')
        plt.bar(year_dict.keys(), year_dict.values())
        plt.show()

        

# [6] Schultz W. Behavioral theories and the neurophysiology of reward. Annual review of psychology. 2006;57:87-115. 
def parse_segment(reference_str):
    """
    Parse a string corresponding to a single article reference
    """
    assert type(reference_str) == str, "input must be string type"
    assert len(reference_str) > 0, "input must contain something"

    # Match the reference string
    regex = re.compile('\[(\d+)\]\s*([^\.]+)?\.([^\.]+)?[\.|\?]([^\.]+)[;\.]*.*?(1[6-9]\d\d|20\d\d)+[;\.]*.*')
    match = regex.match(reference_str)
    #print reference_str
    #assert match != None, "the given reference string cannot match the pattern"
    if match == None:
        print ">>>------------------------"
        print ">>> the given reference string cannot match the pattern"
        print reference_str
        print ">>>------------------------"
        return None

    grps = match.groups()    
    #print grps

    # Create a new Article object
    assert len(grps) >= 4, "segment parsing error, require at least 4 fields"
    ref_id = int(grps[0])
    author_list = grps[1].split(', ')
    title = grps[2].strip()
    venue = grps[3].strip()
    year = int(grps[4]) if len(grps) >= 5 else -1
    return Article(ref_id, author_list, title, venue, year)    
        

def parse_reference_file(fname): 
    """
    Parse a file containing only referneces
    
    A reference might be split in several lines, so we have to 
    put them back into the same line
    """

    fin = open(fname, 'rb')

    # A line of reference in the format
    # [ID] <Authors>. <Title>. <Venue>. <Date>.
    regex = re.compile('\s*\[\d+\]') 
    segment = ""
    article_list = []
    
    for line in fin:
        if regex.match(line) and len(segment) > 0:
            curr_article = parse_segment(segment)
            if curr_article:
                article_list += [ curr_article ]
            segment = line.rstrip('\n')
        else:            
            segment += line.rstrip('\n')

    curr_article = parse_segment(segment)
    if curr_article != None:
        article_list += [ curr_article ]

    return article_list
        


ref_seg0 = "[6] Schultz W. Behavioral theories and the neurophysiology of reward. Annual review of psychology. 2006;57:87-115."
ref_seg1 = "[53] Varela FJ, Thompson E, Rosch E. The embodied mind: Cognitive science and human experience: The MIT Press. 1992."
ref_seg2 = "[16] P. N. Krivitsky, M. S. Handcock, A. E. Raftery, and P. D. Hoff. Representing degree distributions, clustering, and homophily in social networks with latent cluster random effects models. Social Networks, 31(3):204–213, July 2009."
ref_seg3 = "[19] D. Liben-Nowell and J. Kleinberg. The link prediction problem for social networks. In Proceedings of the twelfth international conference on Information and knowledge management, CIKM ’03, pages 556–559, New York, NY, USA, 2003. ACM."

ref_text = """ \
[16] P. N. Krivitsky, M. S. Handcock, A. E. Raftery, and P. D. Hoff. Representing degree distributions, clustering, and homophily in social networks with latent cluster random effects models. Social Networks, 31(3):204–213, July 2009.
[17] A. Lancichinetti and S. Fortunato. Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities. Physical Review E, 80(1):016118, July 2009.
[18] J. Leskovec, K. J. Lang, A. Dasgupta, and M. W. Mahone. Community structure in large networks: Natural cluster sizes and the absence of large well-defined cluster. In Internet Mathematics, 2008.
[19] D. Liben-Nowell and J. Kleinberg. The link prediction problem for social networks. In Proceedings of the twelfth international conference on Information and knowledge management, CIKM ’03, pages 556–559, New York, NY, USA, 2003. ACM.
Duyn, J.H., van Gelderen, P., Li, T.Q., de Zwart, J.A., Koretsky, A.P., Fukunaga, M., 2007. High-field MRI of brain cortical substructure based on signal phase. Proc. Natl. Acad. Sci. U. S. A. 104 (28), 11796–11801.
"""

# Testing the segment parsing function
ref_text = parse_segment(ref_seg2)

# ref_text_list = pdref.split_refs(ref_text)
# article_list = []
# for ref_text in ref_text_list:
#     article_list += [ parse_segment(ref_text) ]
# exit(0)

#article_list = parse_reference_file("draft_references.txt")
# article_list = parse_reference_file("1304.ref.txt")
# article_db = ArticleDB(article_list)
# for art in article_db.article_list:
#     print str(art) + "\n======\n"
#     assert art.year <= 2014, "wow, forward time jumping"
# #article_db.get_authors()
# article_db.get_years()

#for article in article_list:
#    print article.ref_id, article.author_list


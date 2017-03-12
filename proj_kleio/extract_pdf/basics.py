
import re

import references as pdref
import nltk

def utest_pdref():
    refs = pdref.split_refs('''
    [1] J. S. Al-Sumait, J. K. Sykulski, and A. K. Al-Othman, "Solution of different types of economic load dispatch problems using a pattern search method," Electric Power Components and Systems, vol. 36, pp. 250-265, 2008. 
    
    [2] J. S. Al-Sumait, A. K. Al-Othman, and J. K. Sykulski, "Application of pattern search method to power system valve-point economic load dispatch," International Journal of Electrical Power and Energy Systems, vol. 29, pp. 720-730, 2007.''')
    
    pdref.tag_ref('''[1] J. S. Al-Sumait, J. K. Sykulski, and A. K. Al-Othman, "Solution of different types of economic load dispatch problems using a pattern search method," Electric Power Components and Systems, vol. 36, pp. 250-265, 2008.''')


ref_seg0 = "[6] Schultz W. Behavioral theories and the neurophysiology of reward. Annual review of psychology. 2006;57:87-115."
ref_seg1 = "[53] Varela FJ, Thompson E, Rosch E. The embodied mind: Cognitive science and human experience: The MIT Press. 1992."
ref_seg2 = "(16) P. N. Krivitsky, M. S. Handcock, A. E. Raftery, and P. D. Hoff. Representing degree distributions, clustering, and homophily in social networks with latent cluster random effects models. Social Networks, 31(3):204–213, July 2009."
ref_seg3 = "Duyn, J.H., van Gelderen, P., Li, T.Q., de Zwart, J.A., Koretsky, A.P., Fukunaga, M., 2007. High-field MRI of brain cortical substructure based on signal phase. Proc. Natl. Acad. Sci. U. S. A. 104 (28), 11796–11801."


def query_crossref( ref_text ):
    # It almost always starts with a list of authors

    # There is the new API from 
    # http://search.labs.crossref.org/help/api
    #
    # It returns a bunch of json items that matches our query.
    # In case our query string is too fuzzy, we request returning k (=3) items
    # and try to match them against the best value. 
    # The trick is, we look for the match in the query string with the 
    # title and the year of publication.
    # The title is usually easier to find by splitting the text by full stop. 
    # The title would divide the text into two parts, with usually the 
    # first containing the list of authors whereas the second venue of publication and so. 
    import urllib, urllib2

    req_url = 'http://search.labs.crossref.org/dois?q=' + urllib.quote(ref_text) + '&page=1&rows=3'
    #req_url = urllib.encode(ref_seg3)
    req = urllib2.Request(req_url)
    resp = urllib2.urlopen(req)
    resp_html = resp.read()
    print resp_html
    return resp_html


#toks = nltk.word_tokenize(ref_seg0)
#print toks

# # Remove the possible proceeding reference ID
# regex = re.compile('\s*(\[\d+\])?\s*(.*)') 
# print regex.match(ref_seg1).groups()
# print regex.match(ref_seg2).groups()
# print regex.match(ref_seg3).groups()


ref_text = ref_seg2

# Remove the possible proceeding reference ID
regex = re.compile('\s*(\[\d+\]|\(\d+\))?\s*(.*)') 
mg = regex.match(ref_text).groups()
assert len(mg) == 2, "failed to match the reference text"
ref_text = mg[1]

toks = ref_text.split('.')
toks_len = np.array( map(len, toks) )
midx = np.argmax( toks_len )
mlen = toks_len[midx]

title_shft = np.sum(toks_len[:midx]) + midx
authors_raw = ref_text[ : title_shft ]
title_raw = ref_text[ title_shft : title_shft + mlen ]
journal_raw = ref_text[ title_shft + mlen + 1 : ]

print "======================================"
print "AUTHOR :", authors_raw.strip()
print "TITLE  :", title_raw.strip()
print "JOURNAL:", journal_raw.strip()


from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

with open('sample.bib', 'rb') as bibfile:
    bp = BibTexParser(bibfile, customization = convert_to_unicode)
    for ent in bp.get_entry_list():
        print ent

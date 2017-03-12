#!/bin/bash

script=`basename $0`

function usage()
{
    echo "Usage: $script <file.pdf>"
    exit
}

if [[ $# -lt 1 ]]; then
    echo "Error: must provide the pdf file name"
    echo 
    usage
fi
fpath=$1

# http://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html#Shell-Parameter-Expansion
ext=${fpath##*.}
if [ ! -r $fpath ] || [ $ext != "pdf" ]; then
    echo "Error: must provide a valid pdf file with '.pdf' extension"
    usage
fi

fname=`basename $fpath`
fname=${fname%.*}

if [ ! -z `which pdf-extract` ]; then
    # If the CrossRef lab's pdf-extract version exists
    # This is the simplest way to extract references for various files
    # Installation:
    #   gem install pdf-reader -v 1.1.1
    #   gem install pdf-extract
    # 
    # Notice that the result might not always be correct.
    # Their algorithm does not consider a very important structural feature of
    # A bibliography entry must be cited somewhere in the text. 
    # 
    # References:
    #   http://labs.crossref.org/pdfextract/
    #   https://github.com/CrossRef/pdfextract/tree/master/lib/references    
    pdf-extract extract --references $fpath | tee $fname.ref.xml
else
    if [ ! -r $fname.txt ]; then
	java -jar pdfbox-app-1.8.3.jar ExtractText -force -sort $fpath $fname.txt 
    fi
    # This is a heuristic: 
    #   1. -A 7 allows printing seven lines after each matched lines
    #      which we assume is enough for any reasonable bibliography entry
    #   2. We rely on grep to print unique lines and that the paper has
    #      a well formated reference section 
    #
    grep -Ei -A 7 "^\[[0-9]+\].+$" $fname.txt | tee $fname.ref.txt
fi

# Usually the paper content might contain doi link to its website version, 
# at least for the neuroscience papers. NIPS papers are usually parsimonious
# in terms of extra information. Yet this is not a big problem since they
# have a rather consistent naming convention. 


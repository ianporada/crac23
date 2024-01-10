import collections
import json
import logging
import os
import pathlib
import re
from pathlib import Path

from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.oldcorefud import OldCorefUD as OldConlluWriter

logger = logging.getLogger(__name__)



def read_udapi_conllu(fname):
    return ConlluReader(files=fname, split_docs=True).read_documents()


def write_old_conllu(fname, docs):
    fd = open(fname, 'wt', encoding='utf-8')
    writer = OldConlluWriter(filehandle=fd)
    for doc in docs:
        writer.before_process_document(doc)
        writer.process_document(doc)
    fd.flush()
    fd.close()


def convert_to_old_conllu(conllu_files, output_dir):
    for in_fname in conllu_files:
        basename = os.path.basename(in_fname)
        out_fname = os.path.join(output_dir, basename)
        
        if os.path.exists(out_fname):
            return
        
        udapi_docs = read_udapi_conllu(str(in_fname))
        
        write_old_conllu(out_fname, udapi_docs)


def main():
    logging.basicConfig(level=logging.INFO)
    
    base_dir = '/Users/ianporada/Documents/data/crac23/output/'
    # inputs
    input_dir = os.path.join(base_dir, 'dev-fixed/')
    # output
    output_dir = os.path.join(base_dir, 'old_conllu/')
    
    conllu_files = Path(input_dir).glob('en_*.conllu')
            
    convert_to_old_conllu(sorted(conllu_files), output_dir)
        
    return


if __name__ == "__main__":
    main()

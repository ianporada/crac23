import collections
import json
import logging
import os
import pathlib
import re
import subprocess
from pathlib import Path

import udapi
import udapi.core
from udapi.block.corefud.movehead import MoveHead
from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.conllu import Conllu as ConlluWriter
from udapi.core.coref import CorefEntity

logger = logging.getLogger(__name__)


def read_udapi_conllu(fname):
    return ConlluReader(files=fname, split_docs=True).read_documents()


def write_udapi_conllu(fname, docs):
    fd = open(fname, 'wt', encoding='utf-8')
    writer = ConlluWriter(filehandle=fd)
    for doc in docs:
        writer.before_process_document(doc)
        writer.process_document(doc)
    fd.flush()
    fd.close()
        
        
def read_jsonl(fname):
    with open(fname, 'r') as f:
        raw_examples = [json.loads(jline) for jline in f]
    return raw_examples


def evaluate_coreud(gold_path, pred_path):
    cmd = ["python", "../corefud-scorer/corefud-scorer.py", gold_path, pred_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)
    logger.info("Official result for {}".format(pred_path))
    logger.info(stdout)

    cmd = ["python", "../corefud-scorer/corefud-scorer.py", gold_path, pred_path, "-s"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)
    logger.info("Official result with singletons for {}".format(pred_path))
    logger.info(stdout)


def write_udapi_est(blank_fname, est_dir, raw_outputs):
    basename     = os.path.basename(blank_fname)
    dataset_name = os.path.splitext(basename)[0]
    output_fname = os.path.join(est_dir, basename)
    
    if os.path.exists(output_fname):
        return
    
    docs = read_udapi_conllu(blank_fname)
    move_head = MoveHead()
    entity_number = 0
    
    for doc in docs:
        doc_name = doc.meta['docname']
        id = f'{dataset_name}|{doc_name}'
        
        outputs = [x for x in raw_outputs if x['doc_key'] == id]
        assert len(outputs) == 1
        raw_output = outputs[0]
        clusters = raw_output['predicted_clusters']
        subtoken_map = raw_output['orig_subtoken_map']
        
        udapi_words = [word for word in doc.nodes_and_empty]
        doc._eid_to_entity = {}
        
        for cluster in clusters:
            entity_number += 1
            entity = doc.create_coref_entity(eid=f'e{entity_number}')
            for start, end in cluster:
                start_word_idx = subtoken_map[start]
                end_word_idx = subtoken_map[end]
                entity.create_mention(words=udapi_words[start_word_idx: end_word_idx + 1])
            move_head.run(doc)
            udapi.core.coref.store_coref_to_misc(doc)

    write_udapi_conllu(output_fname, docs)


def main():
    # logging.basicConfig(level=logging.WARN)
    
    base_dir = '/Users/ianporada/Documents/data/crac23/'
    
    input_dir  = os.path.join(base_dir, 'input/')
    output_dir = os.path.join(base_dir, 'output/')
    
    for split in ['dev', 'test']:
        blank_dir = os.path.join(input_dir, f'{split}-blind/')
        est_dir   = os.path.join(output_dir, f'{split}-est/')
        
        raw_fname = os.path.join(output_dir, 'raw/', f'{split}.log.jsonl')
        raw_outputs = read_jsonl(raw_fname)
        
        pathlib.Path(est_dir).mkdir(parents=True, exist_ok=True)
        
        conllu_files = Path(blank_dir).glob('*.conllu')
        for blank_fname in sorted(conllu_files):
            write_udapi_est(str(blank_fname), est_dir, raw_outputs)
            
        
    return


if __name__ == "__main__":
    main()

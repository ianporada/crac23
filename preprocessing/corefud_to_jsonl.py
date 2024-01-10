import collections
import json
import logging
import os
import pathlib
import re
from pathlib import Path

from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.conllu import Conllu as ConlluWriter

logger = logging.getLogger(__name__)

RE_SENT_ID = re.compile(r'^# sent_id\s*=?\s*(\S+)')
RE_SPEAKER = re.compile(r'^# speaker\s*=?\s*(\S+)')


def compute_speakers(fname):
    """Given conllu file compute map of sent_id to speaker."""
    sent_to_speaker = {}
    with open(fname, 'r') as file:
        sent_id = None
        for line in file:
            line = line.rstrip()
            if line and line[0] == '#':
                sent_id_match = RE_SENT_ID.match(line)
                if sent_id_match is not None:
                    sent_id = sent_id_match.group(1)
                    continue
                
                speaker_match = RE_SPEAKER.match(line)
                if speaker_match is not None:
                    speaker = speaker_match.group(1)
                    if sent_id:
                        sent_to_speaker[sent_id] = speaker
                    else:
                        raise Exception('Found speaker but no sent_id')
            else:
                sent_id = None
    return sent_to_speaker


def read_udapi_conllu(fname):
    return ConlluReader(files=fname, split_docs=True).read_documents()


def write_udapi_conllu(fname, docs):
    writer = ConlluWriter(filehandle=fname)
    for doc in docs:
        writer.before_process_document(doc)
        writer.process_document(doc)
        
        
def read_jsonl(fname):
    with open(fname, 'r') as f:
        raw_examples = [json.loads(jline) for jline in f]
    return raw_examples


def write_jsonl(fname, examples):
    logging.info('Writing file %s', fname)
    with open(fname, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')


def conllu_to_jsonl(fname):
    logging.info('Reading file %s', fname)
    
    basename     = os.path.basename(fname)
    dataset_name = os.path.splitext(basename)[0]
    
    udapi_docs      = read_udapi_conllu(str(fname))
    sent_to_speaker = compute_speakers(fname)
    
    rows = []
    
    for doc in udapi_docs:
        doc_name = doc.meta['docname']
        
        id = f'{dataset_name}|{doc_name}'
        sentences = []
        speakers = []
        
        sent_idx = -1
        word_idx = -1
        coref_stacks = collections.defaultdict(list)
        clusters = collections.defaultdict(list)
        
        last_sent_id = ''
        
        for node in doc.nodes_and_empty:
            
            sent_id = node.root.sent_id
            
            if sent_id != last_sent_id: # node.ord is the index of the node within the sentence
                sent_idx += 1
                word_idx = -1
                last_sent_id = sent_id
                
                speaker = '-'
                if sent_id in sent_to_speaker:
                    speaker = sent_to_speaker[sent_id]
                    
                sentences += [[]]
                speakers += [[speaker]]
            
            word_idx += 1
            sentences[-1].append(node.form) # add word form to list of words
            
            for mention in set(node.coref_mentions):
                # TODO: think about how to convert zero anaphora
                # TODO: in the catalan corpora, zero-anaphora is never closed and doesn't get read
                # TODO: potentially merge clusters
                
                if ',' in mention.span:
                    continue # skip discontinuous mentions
                
                cluster_id = mention.entity.eid
                mention_start = mention.span.split('-')[0]
                mention_end   = mention.span.split('-')[-1]
                
                if mention_start == str(node.ord): # mention starts at the current node
                    if mention_end == str(node.ord):
                        clusters[cluster_id].append((sent_idx, word_idx, word_idx + 1))
                    else:
                        coref_stacks[cluster_id].append((sent_idx, node.ord, word_idx))
                elif mention_end == str(node.ord):
                    prev_sent_idx, prev_ord, prev_word_idx = coref_stacks[cluster_id].pop()
                    assert prev_sent_idx == sent_idx # make sure mention start and end are in same sentence
                    clusters[cluster_id].append((sent_idx, prev_word_idx, word_idx + 1))
                    
        # TODO: make sure coref stack is empty
                
        # convert clusters to mention clusters
        mention_clusters = list(clusters.values())
        
        assert len(sentences) == len(speakers)
    
        ex = {
            'id': id, # doc id
            'sentences': sentences, # [[w1, ...], ...] of size num_sents x num_words 
            'speakers': speakers, # [speaker1, ...] of size num_sents
            'mention_clusters': mention_clusters, # (sentence, start, end) where [start, end)
            'raw_dataset_name': dataset_name,
            'raw_doc_name': doc_name,
        }
        rows.append(ex)
        
    return rows


def convert_dataset_to_jsonl(conllu_file, jsonl_dir):
    basename     = os.path.basename(conllu_file)
    dataset_name = os.path.splitext(basename)[0]
    
    individual_dir = jsonl_fname  = os.path.join(jsonl_dir, 'individual/')
    jsonl_fname    = os.path.join(individual_dir, f'{dataset_name}.jsonl')
    
    if os.path.exists(jsonl_fname):
        return read_jsonl(jsonl_fname)
    
    pathlib.Path(individual_dir).mkdir(parents=True, exist_ok=True)
    
    rows = conllu_to_jsonl(conllu_file)
    write_jsonl(jsonl_fname, rows)
    
    return rows


def convert_split_to_jsonl(split, conllu_files, jsonl_dir):
    jsonl_fname = os.path.join(jsonl_dir, f'{split}.jsonl')
    if os.path.exists(jsonl_fname):
        return
    
    all_rows = []
    for file in conllu_files:
        all_rows += convert_dataset_to_jsonl(file, jsonl_dir)
    
    write_jsonl(jsonl_fname, all_rows)


def main():
    logging.basicConfig(level=logging.INFO)
    
    base_dir = '/Users/ianporada/Documents/data/crac23/input/'
    # inputs
    train_dir = os.path.join(base_dir, 'CorefUD-1.1-public/', 'data/')
    test_dir  = os.path.join(base_dir, 'test-blind/')
    # output
    jsonl_dir = os.path.join(base_dir, 'jsonl/')
    
    for split in ['dev', 'train', 'test']:
        if split == 'test':
            conllu_files = Path(test_dir).glob('*.conllu')
        else:
            conllu_files = Path(train_dir).glob(f'*/*-{split}.conllu')
            
        convert_split_to_jsonl(split, sorted(conllu_files), jsonl_dir)
        
    return


if __name__ == "__main__":
    main()

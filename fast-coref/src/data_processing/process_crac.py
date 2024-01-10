import collections
import json
import re
from os import path

from coref_utils import conll
from data_processing.constants import SPEAKER_END, SPEAKER_START
from data_processing.utils import (BaseDocumentState, flatten,
                                   get_sentence_map, normalize_word,
                                   parse_args, split_into_segments)


class CracDocumentState(BaseDocumentState):
    def __init__(self, key):
        super().__init__(key)
        self.clusters = collections.defaultdict(list)

    def final_processing(self):
        # merge clusters
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print('Merging clusters (should not happen very often).')
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        self.merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = flatten(merged_clusters)
        self.sentence_map = get_sentence_map(self.segments, self.sentence_end)
        self.subtoken_map = flatten(self.segment_subtoken_map)
        assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(self.subtoken_map), (num_words, len(self.subtoken_map))
        assert num_words == len(self.sentence_map), (num_words, len(self.sentence_map))

    def finalize(self):
        self.final_processing()
        num_words = len(flatten(self.segments))
        assert num_words == len(self.orig_subtoken_map), (
            num_words,
            len(self.orig_subtoken_map),
        )
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "clusters": self.merged_clusters,
            "sentence_map": self.sentence_map,
            "subtoken_map": self.subtoken_map,
            "orig_subtoken_map": self.orig_subtoken_map,
            "orig_tokens": self.tokens,
        }


def process_speaker(speaker):
    speaker = speaker[0]
    speaker = speaker.replace("_", " ")
    return (" ".join([token.capitalize() for token in speaker.split()])).strip()


def get_document(instance, args):
    document_state = CracDocumentState(instance['id'])

    tokenizer = args.tokenizer
    word_idx = -1
    orig_word_idx = -1
    last_speaker = '-'
    speakers = instance['speakers']
    
    sentence_word_map = {} # for mapping preco-style clusters
    
    for sentence_idx, sentence in enumerate(instance['sentences']):
        sentence_word_map[sentence_idx] = {}
        for local_word_idx, word_str in enumerate(sentence):
            
            if args.add_speaker:
                speaker = speakers[sentence_idx]
                if speaker != last_speaker:
                    word_idx += 1
                    # Insert speaker tokens
                    speaker_str = process_speaker(speaker)
                    document_state.tokens.extend(
                        [SPEAKER_START, speaker_str, SPEAKER_END]
                    )
                    speaker_subtokens = []
                    speaker_subtokens.extend(
                        tokenizer.convert_tokens_to_ids([SPEAKER_START])
                    )
                    speaker_subtokens.extend(
                        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(speaker_str))
                    ),
                    speaker_subtokens.extend(
                        tokenizer.convert_tokens_to_ids([SPEAKER_END])
                    )

                    document_state.token_end += (
                        [False] * (len(speaker_subtokens) - 1)
                    ) + [True]
                    for sidx, subtoken in enumerate(speaker_subtokens):
                        document_state.subtokens.append(subtoken)
                        document_state.info.append(None)
                        document_state.sentence_end.append(False)
                        document_state.subtoken_map.append(word_idx)
                        document_state.orig_subtoken_map.append(-1)

                    last_speaker = speaker

            word_idx += 1
            orig_word_idx += 1
            
            # start of map
            sentence_word_map[sentence_idx][local_word_idx] = [len(document_state.subtokens)]
            
            word = normalize_word(word_str)
            subtokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            
            document_state.tokens.append(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]

            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
                document_state.orig_subtoken_map.append(orig_word_idx)
            
            # end of map
            sentence_word_map[sentence_idx][local_word_idx].append(
                len(document_state.subtokens)
            )
            
            # end of sentence
            if local_word_idx == len(sentence) - 1:
                document_state.sentence_end[-1] = True
                
    
    # Map preco-style clusters
    mapped_clusters = []
    for cluster in instance['mention_clusters']:
        cur_cluster = []
        for (sent_idx, word_start, word_end) in cluster:
            span_start = sentence_word_map[sent_idx][word_start][0]
            span_end = sentence_word_map[sent_idx][word_end - 1][1] - 1
            cur_cluster.append((span_start, span_end))
        mapped_clusters.append(sorted(cur_cluster, key=lambda x: x[0]))

    for cluster_i, cluster in enumerate(mapped_clusters):
        document_state.clusters[cluster_i] = cluster

    split_into_segments(
        document_state,
        args.seg_len,
        document_state.sentence_end,
        document_state.token_end,
    )
    document = document_state.finalize()
    return document


def minimize_partition(split, args):
    input_path = path.join(args.input_dir, "{}.jsonl".format(split))
    output_path = path.join(
        args.output_dir, "{}.{}.jsonlines".format(split, args.seg_len)
    )
    count = 0
    print("Minimizing {}".format(input_path))
    
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            instance = json.loads(line.strip())
            documents.append(instance)
            
    with open(output_path, "w") as output_file:
        for instance in documents:
            document = get_document(instance, args)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_split(args):
    tokenizer = args.tokenizer
    if args.add_speaker:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [SPEAKER_START, SPEAKER_END]}
        )

    for split in ["dev", "test", "train"]:
        minimize_partition(split, args)


if __name__ == "__main__":
    minimize_split(parse_args())

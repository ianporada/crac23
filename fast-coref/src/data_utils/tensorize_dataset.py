import collections
import logging
import math
from typing import Dict, List, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


class TensorizeDataset:
    lang_to_prefix = {None: [259, 267, 267, 267],
                      'ca': [960, 259, 267, 267],
                      'cs': [317, 263, 259, 267],
                      'en': [289, 259, 267, 267],
                      'fr': [8967, 259, 267, 267],
                      'de': [269, 259, 267, 267],
                      'hu': [5607, 259, 267, 267],
                      'lt': [259, 2206, 259, 267],
                      'no': [375, 259, 267, 267],
                      'pl': [10440, 259, 267, 267],
                      'ru': [6079, 259, 267, 267],
                      'es': [655, 259, 267, 267],
                      'tr': [534, 259, 267, 267]}
    
    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, remove_singletons: bool = False
    ) -> None:
        self.tokenizer = tokenizer
        self.remove_singletons = remove_singletons
        self.device = torch.device("cpu")

    def tensorize_data(
        self, split_data: List[Dict], training: bool = False
    ) -> List[Dict]:
        tensorized_data = []
        
        dataset_to_outputs = collections.defaultdict(list)
        
        for document in split_data:
            output_dict = self.tensorize_instance_independent(document, training=training)
            
            # rebalance data
            # TODO: fix
            if training and output_dict['doc_key']:
                doc_key = output_dict['doc_key']
                dataset_id, doc_id = doc_key.split('|')
                dataset_to_outputs[dataset_id].append(output_dict)
            
            tensorized_data.append(
                output_dict
            )
        
        # upsample rare datasets
        
        for dataset, outputs in dataset_to_outputs.items():
            n = len(outputs)
            repetitions_to_add = 0
            if n < 50: # small, upsample to 250
                repetitions_to_add = math.floor(250.0 / n) - 1
            elif n < 200: # medium, upsample to 500
                repetitions_to_add = math.floor(500.0 / n) - 1
            elif n <= 500: # medium large, upsample to 1000
                repetitions_to_add = math.floor(1000.0 / n) - 1
                
            for _ in range(repetitions_to_add):
                tensorized_data += outputs
        
        return tensorized_data
    

    def process_segment(self, segment: List, language: str) -> List:
        if self.tokenizer.cls_token_id is None:
            # logging.info('Using T5 prefix.')
            return self.lang_to_prefix[language] + segment + [self.tokenizer.eos_token_id]
            # return [self.tokenizer.convert_tokens_to_ids('<extra_id_0>')] + segment + [self.tokenizer.eos_token_id]
        return [self.tokenizer.cls_token_id] + segment + [self.tokenizer.sep_token_id]

    def tensorize_instance_independent(
        self, document: Dict, training: bool = False
    ) -> Dict:
        segments: List[List[int]] = document["sentences"]
        clusters: List = document.get("clusters", [])
        sentence_map: List[int] = document["sentence_map"]
        subtoken_map: List[int] = document["subtoken_map"]
        
        lang = None
        doc_key = document.get("doc_key", None)
        if doc_key and len(doc_key) >= 2:
            lang = doc_key[:2]

        tensorized_sent: List[Tensor] = [
            torch.unsqueeze(
                torch.tensor(self.process_segment(sent, lang), device=self.device), dim=0
            )
            for sent in segments
        ]

        sent_len_list = [len(sent) for sent in segments]
        output_dict = {
            "tensorized_sent": tensorized_sent,
            "sentences": segments,
            "sent_len_list": sent_len_list,
            "doc_key": document.get("doc_key", None),
            "clusters": clusters,
            "subtoken_map": subtoken_map,
            "sentence_map": torch.tensor(sentence_map, device=self.device),
        }

        # Pass along other metadata
        for key in document:
            if key not in output_dict:
                output_dict[key] = document[key]

        if self.remove_singletons:
            output_dict["clusters"] = [
                cluster for cluster in output_dict["clusters"] if len(cluster) > 1
            ]

        return output_dict

from typing import Dict, List

import torch
from omegaconf import DictConfig
from torch import Tensor

from model.document_encoder.base_encoder import BaseDocEncoder
from pytorch_utils.utils import get_sequence_mask


class IndependentDocEncoder(BaseDocEncoder):
    def __init__(self, config: DictConfig):
        super(IndependentDocEncoder, self).__init__(config)

    def forward(self, document: Dict) -> Tensor:
        doc_tens = document["tensorized_sent"]
        if isinstance(doc_tens, list):
            doc_tens = torch.tensor(doc_tens, device=self.device)
        else:
            doc_tens = doc_tens.to(self.device)

        sent_len_list: List[int] = document["sent_len_list"]
        num_chunks = len(sent_len_list)
        if num_chunks == 1:
            attn_mask = None
        else:
            attn_mask = get_sequence_mask(
                torch.tensor(sent_len_list, device=self.device)
            )

        if not self.config.finetune:
            with torch.no_grad():
                outputs = self.lm_encoder(
                    doc_tens, attention_mask=attn_mask
                )  # C x L x E
        else:
            outputs = self.lm_encoder(doc_tens, attention_mask=attn_mask)  # C x L x E

        encoded_repr = outputs[0]

        unpadded_encoded_output = []

        # TODO: MT5
        
        for idx, sent_len in enumerate(sent_len_list):
            # TODO: handle t5 which has prefix
            model_str: str = self.config.transformer.model_str
            if 'mt5' in model_str:
                unpadded_encoded_output.append(encoded_repr[idx, 4 : sent_len + 4, :])
            else:
                unpadded_encoded_output.append(encoded_repr[idx, 1 : sent_len + 1, :])

        encoded_output = torch.cat(unpadded_encoded_output, dim=0)

        return encoded_output

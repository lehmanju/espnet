#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, Dict, List

import torch

from espnet2.speechlm.predictor.abs_predictor import AbsPredictor

class ParallelPredictor(AbsPredictor):
    def __init__(
        self,
        vocab_size: List,
        input_dim: int,
        nq: int,
    ):
        super(ParallelPredictor, self).__init__()
        
        self.linear = torch.nn.Linear(
            input_dim, vocab_size * nq, bias=False
        )
        self.nq = nq
    
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor, cache: Dict = None,
    ) -> Tuple[torch.Tensor, Dict]:
        
        output = self.linear(input)
        B, T, D  = output.size()
        output = output.view(B, T * self.nq, D // self.nq)

        output_lengths = input_lengths * self.nq
        return output, output_lengths
    
    def get_lookup_table(self):
        raise ValueError("Cannot share the lookup table as there are multiple")
    
    def organize_target(
        self, 
        target: torch.Tensor, 
        target_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        return target, target_lengths

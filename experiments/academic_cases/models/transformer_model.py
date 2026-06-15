import copy

import logging

from typing import Literal

import torch
from torch import nn

from noether.modeling.modules.layers import LinearProjection, RopeFrequency
from noether.modeling.modules import TransformerBlock

from noether.core.schemas import ModelBaseConfig
from noether.core.schemas.modules import LinearProjectionConfig, TransformerBlockConfig
from noether.core.schemas.modules import RopeFrequencyConfig
from noether.core.schemas.modules import LinearProjectionConfig
from noether.core.models import Model

from .attention import BiasDotProductAttention
from academic_cases.datasets import AcademicDataSpecs

logger = logging.getLogger(__name__)

class TransformerModelConfig(ModelBaseConfig):

    name: Literal['transformer']

    data_specs: AcademicDataSpecs

    hidden_dim: int

    depth: int
    transformer_config: TransformerBlockConfig

class TransformerModel(Model):
    '''
    Simple transformer model:
        - linear projection to encode features
        - stack of transformers with RoPE, anchor attention, and optional decomposable bias terms
        - linear projection to decode predictions
    '''

    def __init__(self, config: TransformerModelConfig, 
                 is_frozen = False, 
                 update_counter = None, 
                 path_provider = None, 
                 data_container = None):
        super().__init__(config, is_frozen, update_counter, path_provider, data_container)

        self.rope = RopeFrequency(
            config=RopeFrequencyConfig(
                hidden_dim=config.transformer_config.hidden_dim // config.transformer_config.num_heads,
                input_dim=config.data_specs.position_dim,
                implementation="complex",
            ) 
        )

        self.input_emb = LinearProjection(config=LinearProjectionConfig(
            input_dim=config.data_specs.model_input_dim,
            output_dim=config.hidden_dim
        ))

        transformer_config = copy.deepcopy(config.transformer_config)
        transformer_config.attention_constructor = BiasDotProductAttention
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    transformer_config
                )
                for _ in range(config.depth)
            ]
        )

        self.decoder = LinearProjection(config=LinearProjectionConfig(
            input_dim=config.hidden_dim,
            output_dim=config.data_specs.model_output_dim
        ))

        logger.info(f'Initialised transformer with lbda {config.transformer_config.attention_arguments['lbda']}')

    def forward(self,
                input_feature: torch.Tensor,
                position: torch.Tensor):
        '''Forward pass
        
        Args:
            - input_feature: initial input features: Tensor of shape (B, N_pts, D_in)
            - position: coordinates of points. Tensor of shape (B, N_pts, Ndim)
        '''

        attn_kwargs =  {}

        attn_kwargs['coords'] = position
        attn_kwargs['freqs'] = self.rope(position)

        x = self.input_emb(input_feature)
        for block in self.blocks:
            x = x + block(x, attn_kwargs = attn_kwargs)[0] #blocks now return (out, kv_cache)
        pred = self.decoder(x)

        pred_dict = dict(prediction = pred)

        return pred_dict
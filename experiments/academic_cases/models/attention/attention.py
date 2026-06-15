# Implementation of attention with bias terms

from typing import Literal

import einops
import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Tuple

from noether.modeling.functional.rope import rope
from noether.modeling.modules.attention import DotProductAttention
from noether.core.schemas.modules import DotProductAttentionConfig

float_or_float_list = Union[float, list[float]]

def HPE_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        lbda_plus: torch.Tensor, lbda_minus: torch.Tensor, A: float,
        coords_q = torch.Tensor, coords_kv = torch.Tensor,
        **torch_fn_kwargs):
        '''Attention with hyperbolic PE bias'''

        ### Create attention bias elements ###
        #add dim for heads
        bq = einops.repeat(coords_q, 'bs seqlen ndim -> bs n_heads seqlen ndim', n_heads = q.shape[1])
        bk = einops.repeat(coords_kv, 'bs seqlen ndim -> bs n_heads seqlen ndim', n_heads = k.shape[1])

        bq = torch.concat([bq / lbda_minus, -bq / lbda_plus], axis = -1)
        bk = torch.concat([-bk / lbda_minus, bk / lbda_plus], axis = -1)

        # take the exp
        bq = torch.exp(bq)
        bk = torch.exp(bk)

        # Prefactor
        d = q.shape[-1]
        bq = bq * (-1) * A / 2 * d**0.5

        ###    apply bias     ###
        #concatenate q/bq and k,bk
        q = torch.concat([q,bq], axis = -1).type_as(q)
        k = torch.concat([k,bk], axis = -1).type_as(k)
        ##########################

        ### Run attention kernel ###

        # Extend v to match the shape of q,k (requiered because kernels assume they have the same shapes)
        extra_length = bk.shape[-1]
        extention = torch.ones((*v.shape[:-1], extra_length), dtype = v.dtype, device = v.device)
        v = torch.concatenate([v, extention], axis = -1)

        # print('hpe', q.shape, k.shape, v.shape)
        u_ = F.scaled_dot_product_attention(q,k,v, **torch_fn_kwargs)

        #cut away extention to v
        u = u_[:,:,:,:-extra_length].to(q.dtype)

        return u

def L2_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        lbda:torch.Tensor, A: float,
        coords_q = torch.Tensor, coords_kv = torch.Tensor,
        **torch_fn_kwargs):
        '''Attention with L2 PE bias'''

        ### Create attention bias elements ###
        #add dim for heads
        bq = einops.repeat(coords_q, 'bs seqlen ndim -> bs n_heads seqlen ndim', n_heads = q.shape[1])
        bk = einops.repeat(coords_kv, 'bs seqlen ndim -> bs n_heads seqlen ndim', n_heads = k.shape[1])

        # lbda factor
        bq = bq / lbda
        bk = bk / lbda

        ONE_bq = torch.ones_like(bq)
        ONE_bk = torch.ones_like(bk)
        bq = torch.stack([ bq**2 , - 2 * bq , ONE_bq], axis = -1)
        bk = torch.stack([ ONE_bk , bk  , bk**2], axis = -1)

        # Prefactor
        d = q.shape[-1]
        bq = bq * (-1) * A / 2 * d**0.5

        bq = torch.flatten(bq, -2, -1)
        bk = torch.flatten(bk, -2, -1)

        ###    apply bias     ###
        #concatenate q/bq and k,bk
        q = torch.concat([q,bq], axis = -1).type_as(q)
        k = torch.concat([k,bk], axis = -1).type_as(k)
        ##########################

        ### Run attention kernel ###

        # Extend v to match the shape of q,k (requiered because kernels assume they have the same shapes)
        extra_length = bk.shape[-1]
        extention = torch.ones((*v.shape[:-1], extra_length), dtype = v.dtype, device = v.device)
        v = torch.concatenate([v, extention], axis = -1)

        u_ = F.scaled_dot_product_attention(q,k,v, **torch_fn_kwargs)

        #cut away extention to v
        u = u_[:,:,:,:-extra_length].to(q.dtype)

        return u

class BiasAttentionConfig(DotProductAttentionConfig):
    """Configuration for attention with bias term"""

    bias_type: None | Literal['hyperbolic', 'l2']
    '''Type of bias to use'''

    lbda: None | float_or_float_list | Tuple[float_or_float_list, float_or_float_list]
    '''Lambda parameter. 
      - None: revers to standard attention
      - Float: assymetric, isotropic mask
      - List of length ndim: anysotropic mask, with each element in the list corresponding to a direction
      - Tuple (of float or list): assymetric mask. Can be isotropic (tuple of floats) or anysotropic (tuple of lists)'''

    A: float = 1
    '''Additional amplitude parameter, not used in the paper'''


class BiasDotProductAttention(DotProductAttention):
    """Attention with decomposable bias term. Modified from Noether's DotProductAttention"""

    def __init__(
        self,
        config: BiasAttentionConfig,
    ):
        """

        Args:
            config: Configuration for the BDotProductAttention module.
        """

        super().__init__(config) #initialises projection, heads, rope,...

        config = BiasAttentionConfig(**config.model_dump())

        self.bias_type = config.bias_type

        if self.bias_type is None:
             assert config.lbda is None, 'Bias turned of but lbda provided'
        elif self.bias_type == 'hyperbolic':
            assert config.lbda is not None, 'Need lbda for hyperbolic bias'
            if isinstance(config.lbda, tuple):
                self.register_buffer('lbda_plus',torch.tensor(config.lbda[0]))
                self.register_buffer('lbda_minus',torch.tensor(config.lbda[1]))
            else:
                lbda = torch.tensor(config.lbda)
                self.register_buffer('lbda_plus',lbda)
                self.register_buffer('lbda_minus',lbda)
            self.A = config.A

        elif self.bias_type == 'l2':
            assert config.lbda is not None, 'Need lbda for l2 bias'
            assert not isinstance(config.lbda, tuple), 'Assymetric bias doesn\'t exist for L2 bias'
            self.register_buffer('lbda', torch.tensor(config.lbda))
            self.A = config.A


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward function of the DotProductAttention module.

        Args:
            x: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
            attn_mask: For causal attention (i.e., no attention over the future token) a attention mask should be provided. Defaults to None.
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys. None if use_rope=False.
            coords: coordinates of sequence for bias. Passed to the attention function

        Returns:
            Returns the output of the attention module.
        """

        qkv_weight = torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)
        qkv_bias = torch.cat([self.q.bias, self.k.bias, self.v.bias], dim=0) if self.q.bias is not None else None
        qkv = F.linear(x, qkv_weight, qkv_bias)

        q, k, v = einops.rearrange(
            qkv,
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            assert freqs is not None
            q = rope(q, freqs=freqs)
            k = rope(k, freqs=freqs)
        else:
            assert freqs is None

        if self.bias_type is None:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        elif self.bias_type == 'hyperbolic':
            x = HPE_attention(
                q, k, v, 
                self.lbda_plus, self.lbda_minus, self.A,
                coords_q = coords, coords_kv = coords, 
                attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )
        elif self.bias_type == 'l2':
            x = L2_attention(
                q, k, v, 
                self.lbda, self.A,
                coords_q = coords, coords_kv = coords, 
                attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )

        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj_dropout(self.proj(x))

        return x
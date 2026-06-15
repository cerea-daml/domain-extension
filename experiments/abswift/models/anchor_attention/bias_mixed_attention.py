#  Implementation of bias terms in mixed attention

from typing import Literal

from collections import defaultdict
from collections.abc import Sequence
from itertools import accumulate

from typing import Union, Tuple

import einops
import torch
import torch.nn.functional as F

from noether.core.schemas.modules.attention import AttentionPattern, TokenSpec
from noether.modeling.functional.rope import rope

from noether.core.schemas.modules.attention import MixedAttentionConfig
from noether.modeling.modules.attention.anchor_attention.mixed import MixedAttention

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

class BiasMixedAttentionConfig(MixedAttentionConfig):
    """Configuration for attention with positional bias"""

    bias_type: None | Literal['hyperbolic', 'l2']
    '''Type of bias to use'''

    lbda: None | float_or_float_list | Tuple[float_or_float_list, float_or_float_list]
    '''Lambda parameter. 
      - None: revers to standard attention
      - Float: assymetric, isotropic mask
      - List of length ndim: anysotropic mask, with each element in the list corresponding to a direction
      - Tuple (of float or list): assymetric mask. Can be isotropic (tuple of floats) or anysotropic (tuple of lists)'''

    A: float = 1
    '''Optional amplitude parameter. Not used in the paper.'''

class BiasMixedAttention(MixedAttention):
    """ Implements mixed attention with positional bias. kv caching is not implemented.
    """

    def __init__(
        self,
        config: BiasMixedAttentionConfig,
    ) -> None:
        """
        Args:
            config: Configuration for the MixedAttention module. See
                :class:`~abswift.models.anchor_attention.BiasMixedAttentionConfig` for the available options.
        """
        super().__init__(config=config)

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

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        token_specs: Sequence[TokenSpec],
        attention_patterns: Sequence[AttentionPattern],
        attention_mask: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        kv_cache: None = None
    ) -> torch.Tensor:
        """Apply mixed attention with flexible token-name-based patterns + decomposable bias.

        Args:
            x: Input tensor [batch_size, n_tokens, dim]
            token_specs: Sequence of token specifications defining the input structure: assumes that the input
                x is a concatenation of tokens in the order of token_specs.
            attention_patterns: Sequence of attention patterns to apply. Each pattern defines which
                token groups (queries) attend to which other token groups (keys/values).
                The provided patterns must be exhaustive and non-overlapping. This means every
                token group defined in `token_specs` must be a query in exactly one pattern.
            attention_mask: Optional attention mask (not currently supported)
            freqs: RoPE frequencies for positional encoding
            coords: coordinates for computing the bias
        """
        self._validate_inputs(x, token_specs, attention_patterns, attention_mask, freqs, coords, kv_cache)

        qkv_weight = torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)
        qkv_bias = torch.cat([self.q.bias, self.k.bias, self.v.bias], dim=0) if self.q.bias is not None else None
        qkv = F.linear(x, qkv_weight, qkv_bias)

        q, k, v = einops.rearrange(
            qkv, "bs s (three nh hd) -> three bs nh s hd", three=3, nh=self.num_heads
        ).unbind(0)

        if self.use_rope and freqs is not None:
            q, k = rope(q, freqs=freqs), rope(k, freqs=freqs)

        # Prepare token slices and size map helpers for processing the attention patterns
        sizes = [spec.size for spec in token_specs]
        start_indices = [0] + list(accumulate(sizes[:-1]))
        token_slices = {
            s.name: slice(start, s.size + start) for s, start in zip(token_specs, start_indices, strict=False)
        }
        spec_size_map = {spec.name: spec.size for spec in token_specs}

        token_outputs = self._process_pattern_batched(attention_patterns, q, k, v, coords, token_slices, spec_size_map)  # type: ignore[arg-type]

        # Final assembly and output projection
        output_parts = [token_outputs[spec.name] for spec in token_specs]
        output = torch.cat(output_parts, dim=2)
        output = einops.rearrange(output, "bs nh s hd -> bs s (nh hd)")
        return self.proj(output)  # type: ignore[no-any-return]

    def _process_pattern_batched(
        self,
        attention_patterns: Sequence[AttentionPattern],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor | None,
        token_slices: dict[str, slice],
        spec_size_map: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """Efficient mixed attention implementation that batches compatible (same shape) attention patterns."""
        # Group compatible patterns
        pattern_groups: dict[tuple[int, int], list[AttentionPattern]] = defaultdict(list)
        for pattern in attention_patterns:
            query_len = sum(spec_size_map[name] for name in pattern.query_tokens)
            kv_len = sum(spec_size_map[name] for name in pattern.key_value_tokens)
            pattern_groups[(query_len, kv_len)].append(pattern)

        token_outputs: dict[str, torch.Tensor] = {}
        for group in pattern_groups.values():
            # Concatenate sequences for each attention pattern (e.g. multiple queries and multiple keys & values)
            qs = [torch.cat([q[:, :, token_slices[name]] for name in patt.query_tokens], dim=2) for patt in group]
            ks = [torch.cat([k[:, :, token_slices[name]] for name in patt.key_value_tokens], dim=2) for patt in group]
            vs = [torch.cat([v[:, :, token_slices[name]] for name in patt.key_value_tokens], dim=2) for patt in group]
            if coords is not None:
                coords_q = [torch.cat([coords[:, token_slices[name]] for name in patt.query_tokens], dim=1) for patt in group]
                coords_k = [torch.cat([coords[:, token_slices[name]] for name in patt.key_value_tokens], dim=1) for patt in group]
            # Concatenate independent attentions on batch dimension to process in parallel (e.g. multiple modalities)

            q_batched = torch.cat(qs, dim=0)
            k_batched = torch.cat(ks, dim=0)
            v_batched = torch.cat(vs, dim=0)

            if self.bias_type is None:
                output_batched = F.scaled_dot_product_attention(
                    q_batched, k_batched, v_batched, dropout_p=self.dropout if self.training else 0.0
                )
            elif self.bias_type == 'l2':
                coords_q_batched = torch.cat(coords_q, dim = 0)
                coords_k_batched = torch.cat(coords_k, dim = 0)
                output_batched = L2_attention(
                        q_batched, k_batched, v_batched, 
                        self.lbda, self.A,
                        coords_q = coords_q_batched, coords_kv = coords_k_batched, 
                        dropout_p=self.dropout if self.training else 0.0
                    )
            elif self.bias_type == 'hyperbolic':
                coords_q_batched = torch.cat(coords_q, dim = 0)
                coords_k_batched = torch.cat(coords_k, dim = 0)
                output_batched = HPE_attention(
                        q_batched, k_batched, v_batched, 
                        self.lbda_plus, self.lbda_minus, self.A,
                        coords_q = coords_q_batched, coords_kv = coords_k_batched, 
                        dropout_p=self.dropout if self.training else 0.0
                    )
            
            # Undo the batch concatenation
            output_chunks = torch.chunk(output_batched, chunks=len(group), dim=0)
            # Undo the sequence concatenation
            for pattern, pattern_output in zip(group, output_chunks, strict=True):
                query_sizes = [spec_size_map[name] for name in pattern.query_tokens]
                for name, chunk in zip(pattern.query_tokens, pattern_output.split(query_sizes, dim=2), strict=True):
                    token_outputs[name] = chunk
        return token_outputs

    def _validate_inputs(
        self,
        x: torch.Tensor,
        token_specs: Sequence[TokenSpec],
        attention_patterns: Sequence[AttentionPattern],
        attention_mask: torch.Tensor | None,
        freqs: torch.Tensor | None,
        coords: torch.Tensor | None = None,
        kv_cache: None = None
    ) -> None:
        
        """Validate input consistency."""
        if kv_cache is not None:
            raise ValueError("kv_cache is not implemented for BiasMixedAttention and shjould be left as None")
        if not ((self.bias_type is None) == (coords is None)):
            raise ValueError(f"Bias usage mismatch: self.bias_type = {self.bias_type}, but coords is {coords is not None}")

        super()._validate_inputs(
            x, token_specs, attention_patterns, attention_mask, freqs
        )

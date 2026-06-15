# Implementation of the different ancho attention classes, using decomposable bias terms

import abc
from typing import Sequence, Union, Tuple, Literal

from pydantic import Field

import torch

from noether.modeling.modules.attention.anchor_attention import MultiBranchAnchorAttention

from noether.core.schemas.modules.attention import AttentionConfig

from noether.core.schemas.modules.attention import (
    AttentionPattern,
    TokenSpec,
)

from .bias_mixed_attention import BiasMixedAttention, BiasMixedAttentionConfig

#################################################### configs ##################################################
float_or_float_list = Union[float, list[float]]

class BiasMultiBranchAnchorAttentionConfig(AttentionConfig, metaclass=abc.ABCMeta):
    """Configuration for Multi-Branch Anchor Attention module with decomposable bias terms."""

    branches: list[str] = Field(..., min_length=1)
    anchor_suffix: str = Field("_anchors")

    bias_type: None | Literal['hyperbolic', 'l2']
    '''Type of bias to use'''

    lbda: None | float_or_float_list | Tuple[float_or_float_list, float_or_float_list]
    '''Lambda parameter. 
      - None: revers to standard attention
      - Float: assymetric, isotropic mask
      - List of length ndim: anysotropic mask, with each element in the list corresponding to a direction
      - Tuple (of float or list): assymetric mask. Can be isotropic (tuple of floats) or anysotropic (tuple of lists)'''

    A: None | float
    '''A parameter'''

class BiasCrossAnchorAttentionConfig(BiasMultiBranchAnchorAttentionConfig):
    """Configuration for Cross Anchor Attention module."""


class BiasJointAnchorAttentionConfig(BiasMultiBranchAnchorAttentionConfig):
    """Configuration for Joint Anchor Attention module."""


############################################################ attention layers ########################################################

class BiasMultiBranchAnchorAttention(MultiBranchAnchorAttention):
    '''Mixes multi-branch anchor attention and decomposable bias. 
    We only redefine the self.mixed_attention and leave the rest to parent's class'''

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        _config: BiasMultiBranchAnchorAttentionConfig = BiasMultiBranchAnchorAttentionConfig(**config.model_dump())  # type: ignore[no-redef]

        if not _config.branches:
            raise ValueError("The 'branches' list cannot be empty.")

        self.mixed_attention = BiasMixedAttention(
            config=BiasMixedAttentionConfig(
                hidden_dim=_config.hidden_dim,
                num_heads=_config.num_heads,
                use_rope=_config.use_rope,
                bias=_config.bias,
                init_weights=_config.init_weights,
                dropout=_config.dropout,

                bias_type=_config.bias_type,
                lbda = _config.lbda,
                A = _config.A


            )  # type: ignore[call-arg]
        )
        self.branches = _config.branches
        self.anchor_suffix = _config.anchor_suffix

    def forward(
        self,
        x: torch.Tensor,
        token_specs: Sequence[TokenSpec],
        freqs: torch.Tensor | None = None,
        coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply attention using the patterns defined by the subclass."""
        self._validate(token_specs)
        patterns = self._create_attention_patterns(token_specs)
        return self.mixed_attention(x, token_specs, patterns, freqs=freqs, coords = coords)  # type: ignore[no-any-return]


########## Self, Cross, and Joint attention implementation #########
# The code is copy-pasted from respective noether's implementation #
#       only the parent class and config class are changed         #

class BiasCrossAnchorAttention(BiasMultiBranchAnchorAttention):
    """Cross anchor + bias
    """

    def __init__(
        self,
        config: BiasCrossAnchorAttentionConfig,
    ):
        """

        Args:
            config: Configuration for the CrossAnchorAttention module. See
                :class:`~noether.core.schemas.modules.attention.CrossAnchorAttentionConfig` for the available options.
        """
        if len(config.branches) < 2:
            raise ValueError("CrossAnchorAttention requires at least two branches.")
        super().__init__(
            config=config,
        )

    def _create_attention_patterns(self, token_specs: Sequence[TokenSpec]) -> Sequence[AttentionPattern]:
        """Create cross-attention patterns where each branch attends to other branches' anchors."""
        patterns = []
        for query_branch in self.branches:
            query_tokens = [spec.name for spec in token_specs if spec.name.startswith(query_branch)]
            other_branches = [name for name in self.branches if name != query_branch]
            key_value_anchors = [f"{other_branch}{self.anchor_suffix}" for other_branch in other_branches]
            attention_pattern = AttentionPattern(query_tokens=query_tokens, key_value_tokens=key_value_anchors)
            patterns.append(attention_pattern)
        return patterns
    

class BiasJointAnchorAttention(BiasMultiBranchAnchorAttention):
    """ Joint ancor + bias
    """

    def __init__(
        self,
        config: BiasJointAnchorAttentionConfig,
    ):
        if len(config.branches) < 2:
            raise ValueError("JointAnchorAttention requires at least two branches. Otherwise use SelfAnchorAttention.")
        super().__init__(
            config=config,
        )

    def _create_attention_patterns(self, token_specs: Sequence[TokenSpec]) -> Sequence[AttentionPattern]:
        """Create joint attention pattern where all tokens attend to all anchors."""
        all_anchor_tokens = [f"{branch}{self.anchor_suffix}" for branch in self.branches]
        all_query_tokens = [spec.name for spec in token_specs]
        attention_pattern = AttentionPattern(query_tokens=all_query_tokens, key_value_tokens=all_anchor_tokens)
        return [attention_pattern]
    
class BiasSelfAnchorAttention(BiasMultiBranchAnchorAttention):
    """ Self anchor + bias
    """

    def _create_attention_patterns(self, token_specs: Sequence[TokenSpec]) -> Sequence[AttentionPattern]:
        """Create self-attention patterns where each branch attends to its own anchors."""
        patterns = []
        for branch in self.branches:
            anchor_name = f"{branch}{self.anchor_suffix}"
            branch_query_tokens = [spec.name for spec in token_specs if spec.name.startswith(branch)]
            attention_pattern = AttentionPattern(query_tokens=branch_query_tokens, key_value_tokens=[anchor_name])
            patterns.append(attention_pattern)
        return patterns

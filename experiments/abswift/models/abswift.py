# AB-SWIFT implementation using Noether framework

import copy
import logging

from typing import Literal, Union, Tuple
from pydantic import Field

import torch
from torch import nn

from noether.core.schemas import ModelBaseConfig
from noether.core.schemas.modules import TransformerBlockConfig, SupernodePoolingConfig, MLPConfig
from noether.core.types import InitWeightsMode

from noether.core.models import Model
from noether.core.schemas.modules.attention import TokenSpec
from noether.core.schemas.modules.layers import (
    RopeFrequencyConfig,
)

from noether.core.schemas.dataset import ModelDataSpecs

# from noether.modeling.modules.attention.anchor_attention import (
#     CrossAnchorAttention,
#     JointAnchorAttention,
#     SelfAnchorAttention,
# )

from noether.modeling.modules.blocks import TransformerBlock
from noether.modeling.modules.encoders import SupernodePooling
from noether.modeling.modules.layers import RopeFrequency
from noether.modeling.modules.mlp import MLP

import einops

# from abswift.datasets import AtmoDataSpecs

# anchor attention implementations with bias terms
from .anchor_attention import (
    BiasSelfAnchorAttention, 
    BiasCrossAnchorAttention, 
    BiasJointAnchorAttention
)

logger = logging.getLogger(__name__)


#################################### config ######################################

float_or_float_list = Union[float, list[float]]
class ABSWIFTConfig(ModelBaseConfig):
    
    # model_config = ConfigDict(extra="forbid")

    name: Literal['abswift'] #pydandic wasn't satisfied with default str definition, not sure why

    hidden_dim: int = Field(...)
    """Hidden dimension of the model."""

    terrain_supernode_radius: int
    '''Supernode pooling radius for the terrain supernodes'''
    obstacles_supernode_radius: int
    '''Supernode pooling radius for the obstacles supernodes'''

    transformer_block_config: TransformerBlockConfig
    '''Config for the different transformer blocks'''

    geometry_preprocessor_blocks: list[Literal["shared", "cross", "joint"]]
    """Types of geometry blocks to use in the geometry encoder. cf processor blocks for block types, with the two branches being terrain and obstacles"""

    physics_blocks: list[Literal["shared", "cross", "joint"]]
    """Types of physics blocks to use in the model.
    Options are "shared", "cross" and "joint".
    Shared: Self-attention within a branch (geometry or volume). Attention blocks share weights between geometry and volume.
    Cross: Cross-attention between geometry and volume branches. Weights are shared between geometry and volume.
    Joint: Joint attention over geometry and volume points. I.e. full self-attention over both geometry and volume points."""

    num_volume_blocks: int = Field(...)
    """Number of transformer blocks in the volume decoder."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Weight initialization of linear layers. Defaults to "truncnormal002"."""

    drop_path_rate: float = Field(0.0)
    """Drop path rate for stochastic depth. Defaults to 0.0 (no drop path)."""

    data_specs: ModelDataSpecs
    """Data specifications for the model."""

    bias_type: None | Literal['hyperbolic', 'l2']
    '''Type of bias to use. Same for all blocks'''

    A: float = 1
    """Optional amplitude parameter for bias (same for all blocks)"""

    geometry_lbda: None | float_or_float_list | Tuple[float_or_float_list, float_or_float_list] = None
    '''lbda for the geometry. If None, no bias is applied. Defaults to None
      See ~abswift.schemas.model.attention.BiasAttentionConfig for available options'''
    
    processor_lbda: None | float_or_float_list | Tuple[float_or_float_list, float_or_float_list] = None
    '''lbda for the processor. If None, no bias is applied. Defaults to None
      See ~abswift.schemas.model.attention.BiasAttentionConfig for available options'''
    
    decoder_lbda: None | float_or_float_list | Tuple[float_or_float_list, float_or_float_list] = None
    '''lbda for the (volume) decoder. If None, no bias is applied. Defaults to None
      See ~abswift.schemas.model.attention.BiasAttentionConfig for available options'''
   
############################################# core model ########################################

class ABSWIFT_torch(nn.Module):
    '''Anchored Branched Steady-state WInd Flow Transformer
    This implementation follows the paper implementation, but adds bias options 
    and do NOT use absolute position embeddings to generate initial latent embeddings. 
    This makes the model fully equivariant by translation
    '''

    def __init__(
        self,
        config:ABSWIFTConfig,
        **kwargs,):
        super().__init__(**kwargs)

        self.data_specs = config.data_specs

        #Conditioning: we do not use it, as it is not part of the original implementation
        config.transformer_block_config.condition_dim = None

        if not config.transformer_block_config.use_rope:
            raise ValueError("AB-SWIFT requires RoPE to be enabled in the transformer block config.")

        self.rope = RopeFrequency(
            config=RopeFrequencyConfig(
                hidden_dim=config.transformer_block_config.hidden_dim // config.transformer_block_config.num_heads,
                input_dim=config.data_specs.position_dim,
                implementation="complex",
            )  # type: ignore[call-arg]
        )  # type: ignore[call-arg]

        #volume embedding -> We don't embed abs pos anymore
        #self.volume_bias = MLP(
        #    config=MLPConfig(  # type: ignore[call-arg]
        #        input_dim=config.hidden_dim,
        #        hidden_dim=config.hidden_dim,
        #        output_dim=config.hidden_dim,
        #        bias=config.transformer_block_config.bias,
        #    )
        #)

        #Profile embedding
        self.profile_encoder = MLP(
            MLPConfig(
                input_dim=self.data_specs.conditioning_dims['profile'],
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                bias=config.transformer_block_config.bias,
        ))

        # geometry
        self.obstacles_supernode_pooling = SupernodePooling(
            config=SupernodePoolingConfig(
                    hidden_dim=config.hidden_dim,
                    input_dim=self.data_specs.position_dim,
                    radius=config.obstacles_supernode_radius,
                    spool_pos_mode='relpos',
                    readd_supernode_pos=False,
                    input_features_dim=self.data_specs.domains['obstacles'].feature_dim.total_dim or None
            ))
        self.terrain_supernode_pooling = SupernodePooling(
            config=SupernodePoolingConfig(
                    hidden_dim=config.hidden_dim,
                    input_dim=self.data_specs.position_dim,
                    radius=config.terrain_supernode_radius,
                    spool_pos_mode='relpos',
                    readd_supernode_pos=False,
                    input_features_dim=self.data_specs.domains['terrain'].feature_dim.total_dim or None
            ))

        self.geometry_blocks = nn.ModuleList()
        for block in config.geometry_preprocessor_blocks:
            if block == "shared":
                attention_constructor = BiasSelfAnchorAttention  # type: ignore[assignment]
            elif block == "cross":
                attention_constructor = BiasCrossAnchorAttention  # type: ignore[assignment]
            elif block == "joint":
                attention_constructor = BiasJointAnchorAttention  # type: ignore[assignment]
            else:
                raise NotImplementedError(
                    f"Unknown physics block type: {block}. Supported: shared, cross, joint."
                )

            block_config = copy.deepcopy(config.transformer_block_config)
            block_config.attention_constructor = attention_constructor  # type: ignore[assignment]
            block_config.attention_arguments = {"branches": ("terrain", "obstacles"), "bias_type":config.bias_type, "lbda": config.geometry_lbda, "A": config.A}
            block = TransformerBlock(config=block_config)  # type: ignore[assignment]

            self.geometry_blocks.append(block)  # type: ignore[arg-type]

        self.physics_blocks = nn.ModuleList()
        for block in config.physics_blocks:
            if block == "shared":
                attention_constructor = BiasSelfAnchorAttention  # type: ignore[assignment]
            elif block == "cross":
                attention_constructor = BiasCrossAnchorAttention  # type: ignore[assignment]
            elif block == "joint":
                attention_constructor = BiasJointAnchorAttention  # type: ignore[assignment]
            else:
                raise NotImplementedError(
                    f"Unknown physics block type: {block}. Supported: shared, cross, joint."
                )

            block_config = copy.deepcopy(config.transformer_block_config)
            block_config.attention_constructor = attention_constructor  # type: ignore[assignment]
            block_config.attention_arguments = {"branches": ("geometry", "volume"), "bias_type":config.bias_type, "lbda": config.processor_lbda, "A": config.A}
            block = TransformerBlock(config=block_config)  # type: ignore[assignment]

            self.physics_blocks.append(block)  # type: ignore[arg-type]

        # volume decoder blocks
        volume_blocks_config = copy.deepcopy(config.transformer_block_config)
        volume_blocks_config.attention_constructor = BiasSelfAnchorAttention  # type: ignore[assignment]
        volume_blocks_config.attention_arguments = {"branches": ("volume",),  "bias_type":config.bias_type, "lbda": config.decoder_lbda, "A": config.A}
        self.volume_decoder_blocks = nn.ModuleList(
            [TransformerBlock(config=volume_blocks_config) for _ in range(config.num_volume_blocks)],
        )

        #decoder: separate Mlps for each field
        self.decoder = nn.ModuleDict({
            field: MLP(
                config=MLPConfig(  # type: ignore[call-arg]
                    input_dim=config.hidden_dim,
                    hidden_dim= 4 * config.hidden_dim, #expantion factor of 4 following the AB-SWIFT paper
                    output_dim=out_dim,
                    bias=config.transformer_block_config.bias,
                )
            )
            for field, out_dim in config.data_specs.domains['volume'].output_dims.items()
        })

        # init weights
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)

        self._use_bias_in_geom = config.geometry_lbda is not None
        self._use_bias_in_processor = config.processor_lbda is not None
        self._use_bias_in_decoder = config.decoder_lbda is not None

        logger.info('Initialised AB-SWIFT')

    def _split_tensor_by_token_specs(
        self, tensor: torch.Tensor, token_specs: list[TokenSpec]
    ) -> dict[str, torch.Tensor]:
        """Split tensor according to token specifications."""
        sizes = [spec.size for spec in token_specs]
        splits = tensor.split(sizes, dim=1)
        return {spec.name: split for spec, split in zip(token_specs, splits, strict=True)}

    def geometry_branch_forward(
        self,
        obstacles_position: torch.Tensor,
        terrain_position: torch.Tensor,
        obstacles_supernode_idx: torch.Tensor,
        terrain_supernode_idx: torch.Tensor,
        obstacles_batch_idx: torch.Tensor,
        terrain_batch_idx: torch.Tensor,
        obstacles_features: torch.Tensor | None,
        terrain_features: torch.Tensor | None,
        condition: torch.Tensor | None,
        # token_specs: list[torch.Tensor],
        geometry_attn_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the geometry branch of the model.
        """

        # encode obstacles -> yields shape (B, N_supernodes, D)
        obstacles_encoding = self.obstacles_supernode_pooling(
            input_pos=obstacles_position,
            supernode_idx=obstacles_supernode_idx,
            batch_idx=obstacles_batch_idx,
            input_features = obstacles_features
        )
        # encode terrain
        terrain_encoding = self.terrain_supernode_pooling(
            input_pos=terrain_position,
            supernode_idx=terrain_supernode_idx,
            batch_idx=terrain_batch_idx,
            input_features = terrain_features,
        )
        #concatenate both sequences
        geometry_encoding = torch.concat([obstacles_encoding, terrain_encoding], axis = 1)

        #create tokens specs (from sequences properly shaped as (B, N_supernodes, D) )
        obstacles_token_specs = TokenSpec(name="obstacles_anchors", size=obstacles_encoding.size(1))
        terrain_token_specs = TokenSpec(name="terrain_anchors", size=terrain_encoding.size(1))
        token_specs = [obstacles_token_specs, terrain_token_specs]

        #transfomer blocks
        if len(self.geometry_blocks) > 0:
            # feed supernodes through some transformer blocks
            for block in self.geometry_blocks:
                geometry_encoding, _ = block(
                    geometry_encoding,
                    attn_kwargs=dict(token_specs=token_specs, **geometry_attn_kwargs),
                    condition=condition,
                )
        return geometry_encoding
    
    def volume_encoder_forward(
            self,
            volume_position_all: torch.Tensor,
            profile: torch.Tensor,
    ) -> torch.Tensor:
        '''Volume position and profile encoding
        We do not use absolute position embeddings, ie volume embeddings are determined by the profile embeddings and are constant accross all volume points
        args:
            volume_position_all: Tensor of shape (B, N_volume_total, D_pos)
            profile: Tensor of shape (B, N_z_profile, D_variable_profiles)'''

        if not (volume_position_all.ndim == 3):
            raise ValueError("volume_position_all must be a 3-dimensional tensor.")
        
        profile = torch.flatten(profile, start_dim=-2, end_dim=-1)
        x_profile = self.profile_encoder(profile)

        #we set the same profile embeddings for all volume points
        x_volume = einops.repeat(x_profile, 'B D -> B N_vol D', N_vol = volume_position_all.shape[1])

        return x_volume

    def processor_blocks_forward(
        self,
        x_geometry: torch.Tensor,
        x_volume: torch.Tensor,
        processor_token_specs: list[TokenSpec],
        processor_attn_kwargs: dict[str, torch.Tensor],
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forward pass through the physics processor blocks of the model.

        Args:
            x_geometry: Tensor of shape (B, N_geometry, D_hidden)
            x_volume: Tensor of shape (B, N_volume_total, D_hidden)
            physics_token_specs: List of TokenSpec defining the token specifications for the physics blocks.
            physics_attn_kwargs: Additional attention kwargs for the physics transformer blocks.
            condition: Optional conditioning tensor of shape (B, D_condition)
        """

        x_physics = torch.concat([x_geometry, x_volume], dim = 1)
        for block in self.physics_blocks:
            if isinstance(block, TransformerBlock):
                x_physics, _ = block(
                    x_physics,
                    attn_kwargs=dict(token_specs=processor_token_specs, **processor_attn_kwargs),
                    condition=condition,
                )
            else:
                raise NotImplementedError(f"Unknown block type: {type(block)}")

        return x_physics

    def volume_decoder_blocks_forward(
        self,
        x_volume: torch.Tensor,
        volume_token_specs: list[TokenSpec],
        volume_position_all: torch.Tensor,
        volume_decoder_attn_kwargs: dict[str, torch.Tensor],
        condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder blocks of the model.
        """

        if not (x_volume.size(1) == volume_position_all.size(1)):
            raise ValueError("Volume tensor size does not match volume position size.")

        # volume decoder blocks
        for block in self.volume_decoder_blocks:
            x_volume, _ = block(
                x_volume,
                attn_kwargs=dict(token_specs=volume_token_specs, **volume_decoder_attn_kwargs),
                condition=condition,
            )

        #Mlp decoders
        volume_predictions = {field:decoder(x_volume) for field, decoder in self.decoder.items()}

        return volume_predictions

    def create_attn_kwargs(
        self,
        obstacles_position: torch.Tensor,
        terrain_position: torch.Tensor,
        obstacles_supernode_idx: torch.Tensor,
        terrain_supernode_idx: torch.Tensor,
        volume_position_all: torch.Tensor,
    ):
        """Create RoPE frequencies + position tensors for all relevant positions.

        Args:
            obstacles_position: Tensor of shape (B * N_obstacles, D_pos), sparse tensor.
            terrain_position: Tensor of shape (B * N_terrain, D_pos), sparse tensor.
            obstacles_supernode_idx: Tensor of shape (B * N_obstacles_supernodes,) with indices of supernodes
            terrain_supernode_idx: Tensor of shape (B * N_terrain_supernodes,) with indices of supernodes
            volume_position_all: Tensor of shape (B, N_volume_total, D_pos)
        """

        batch_size = volume_position_all.size(0)
        ndim = volume_position_all.size(-1)
        geometry_attn_kwargs = {}
        physics_attn_kwargs = {}
        volume_decoder_attn_kwargs = {}

        # coordinates for bias
        obstacles_sn_pos = obstacles_position[obstacles_supernode_idx].view(batch_size, -1, ndim)
        terrain_sn_pos = terrain_position[terrain_supernode_idx].view(batch_size, -1, ndim)
        geometry_position = torch.concat([obstacles_sn_pos, terrain_sn_pos], dim = 1)
        
        if self._use_bias_in_geom:
            geometry_attn_kwargs['coords'] = geometry_position
        
        geometry_volume_pos = torch.concat([geometry_position, volume_position_all], dim = 1)
        if self._use_bias_in_processor:
            physics_attn_kwargs['coords'] = geometry_volume_pos

        if self._use_bias_in_decoder:        
            volume_decoder_attn_kwargs['coords'] = volume_position_all

        # RoPE frequencies
        geometry_rope = self.rope(geometry_position)
        geometry_attn_kwargs["freqs"] = geometry_rope

        rope_volume_all = self.rope(volume_position_all)
        volume_decoder_attn_kwargs["freqs"] = rope_volume_all

        rope_all = torch.concat([geometry_rope, rope_volume_all], dim=1)
        physics_attn_kwargs["freqs"] = rope_all

        return (
            geometry_attn_kwargs,
            physics_attn_kwargs,
            volume_decoder_attn_kwargs,
        )

    def forward(
        self,
        # geometry
        obstacles_position: torch.Tensor,
        terrain_position: torch.Tensor,
        obstacles_supernode_idx: torch.Tensor,
        terrain_supernode_idx: torch.Tensor,
        obstacles_batch_idx: torch.Tensor | None,
        terrain_batch_idx: torch.Tensor | None,
        # anchors
        anchor_volume_position: torch.Tensor,
        # design parameters
        meteorological_profile: torch.Tensor,
        # optional geometry features
        obstacles_features: torch.Tensor | None = None,
        terrain_features: torch.Tensor | None = None,
        # queries
        query_volume_position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the AB-SWIFT model.
        Args:
            obstacles_position: Coordinates of the obstacles. Tensor of shape (B * N_obstacles, D_pos), sparse tensor
            terrain_position: Coordinates of the terrain. Tensor of shape (B * N_terrain, D_pos), sparse tensor
            obstacles_supernode_idx: Indices of the supernodes for the obstacles points. Tensor of shape (B * N_obstacles_supernodes,)
            terrain_supernode_idx: Indices of the supernodes for the terrain points. Tensor of shape (B * N_terrain_supernodes,)
            obstacles_batch_idx: Batch indices for the obstacles points. Tensor of shape (B * N_obstacles,). If None, assumes all points belong to the same batch.
            terrain_batch_idx: Batch indices for the terrain points. Tensor of shape (B * N_terrain,). If None, assumes all points belong to the same batch.
            volume_anchor_position: Coordinates of the volume anchor points. Tensor of shape (B, N_volume_anchor, D_pos)
            meteorological_profile: Meteorological profile. Tensor of shape (B, N_z_profile, D_variable_profiles)
            obstacles_features: optional features for the obstacles (eg normals or opacity). None or Tensor of shape (B * N_obstacles, D_obstacles_features)
            terrain_features: optional features for the terrain (eg rugosity). None or Tensor of shape (B * N_terrain, D_terrain_features)
            query_volume_position: Coordinates of the query volume points.
        """
        
        condition = None
        
        if query_volume_position is None:
            volume_position_all = anchor_volume_position
        else:
            volume_position_all = torch.concat([anchor_volume_position, query_volume_position], dim=1)

        # rope frequencies and coordinates
        (
            geometry_attn_kwargs,
            processor_attn_kwargs,
            volume_decoder_attn_kwargs,
        ) = self.create_attn_kwargs(
            obstacles_position, 
            terrain_position, 
            obstacles_supernode_idx,
            terrain_supernode_idx,
            volume_position_all
        )
        # geometry branch
        geometry_encoding = self.geometry_branch_forward(
            obstacles_position,
            terrain_position,
            obstacles_supernode_idx,
            terrain_supernode_idx,
            obstacles_batch_idx,
            terrain_batch_idx,
            obstacles_features,
            terrain_features,
            condition,
            # geometry_encoder_token_specs,
            geometry_attn_kwargs
        )

        #volume encoding from profile
        x_volume = self.volume_encoder_forward(
            volume_position_all,
            meteorological_profile
        )

        #create token specs for the processor branch
        processor_token_specs: list[TokenSpec] = []
        geometry_token_specs = [TokenSpec(name="geometry_anchors", size=geometry_encoding.size(1))]
        volume_token_specs = [TokenSpec(name="volume_anchors", size=anchor_volume_position.size(1))]
        if query_volume_position is not None:
            volume_token_specs.append(TokenSpec(name="volume_queries", size=query_volume_position.size(1)))
        processor_token_specs.extend(geometry_token_specs)
        processor_token_specs.extend(volume_token_specs)

        # processor blocks
        x_physics = self.processor_blocks_forward(
            x_geometry = geometry_encoding,
            x_volume = x_volume,
            processor_token_specs = processor_token_specs,
            processor_attn_kwargs = processor_attn_kwargs,
            condition=condition,
        )

        # Split volume tokens from the physics tokens
        token_dict = self._split_tensor_by_token_specs(x_physics, processor_token_specs)
        volume_tensors = [token_dict[spec.name] for spec in processor_token_specs if spec.name.startswith("volume")]
        x_volume = torch.cat(volume_tensors, dim=1)

        # decoder blocks
        volume_predictions = self.volume_decoder_blocks_forward(
            x_volume=x_volume,
            volume_token_specs=volume_token_specs,
            volume_position_all=volume_position_all,
            volume_decoder_attn_kwargs=volume_decoder_attn_kwargs,
            condition=condition,
        )

        # Split beween anchor and query predictions
        predictions = {'anchor_'+k:v[:, : anchor_volume_position.size(1)] for k,v in volume_predictions.items()}
        if query_volume_position is not None:
            volume_query_predictions = {'query_'+k:v[:, anchor_volume_position.size(1) : ] for k,v in volume_predictions.items()}
            predictions = {**predictions, **volume_query_predictions}

        return predictions
    

###################################### Noether wrapper ###################################

class ABSWIFT(Model):
    """Noether wrapper around AB-SWIFT.
    This implementation follows the paper implementation, but adds decomposable bias options 
    and do NOT use absolute position embeddings to generate initial latent embeddings. 
    This makes the model fully invariant by translation, assuming attention embeddings are relative"""

    def __init__(
        self,
        model_config: ABSWIFTConfig,
        **kwargs,
    ):
        """Initialize the AB-SWIFT model.

        Args:
            model_config: The configuration for the AB-SWIFT model.
        """

        super().__init__(model_config=model_config, **kwargs)

        self.ab_swift = ABSWIFT_torch(
            config=model_config,
        )

    def forward(
        self,
        # geometry
        obstacles_position: torch.Tensor,
        terrain_position: torch.Tensor,
        obstacles_supernode_idx: torch.Tensor,
        terrain_supernode_idx: torch.Tensor,
        obstacles_batch_idx: torch.Tensor | None,
        terrain_batch_idx: torch.Tensor | None,
        # anchors
        anchor_volume_position: torch.Tensor,
        # design parameters
        meteorological_profile: torch.Tensor,
        # optional geometry features
        obstacles_features: torch.Tensor | None = None,
        terrain_features: torch.Tensor | None = None,
        # queries
        query_volume_position: torch.Tensor | None = None,
        # **kwargs
    ) -> dict[str, torch.Tensor]:
        
        out = self.ab_swift( #**kwargs
            # geometry
            obstacles_position,
            terrain_position,
            obstacles_supernode_idx,
            terrain_supernode_idx,
            obstacles_batch_idx,
            terrain_batch_idx,
            # anchors
            anchor_volume_position,
            # design parameters
            meteorological_profile,
            # optional geometry features
            obstacles_features,
            terrain_features,
            # queries
            query_volume_position,
        )

        return out

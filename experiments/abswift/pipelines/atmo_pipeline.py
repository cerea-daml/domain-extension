#Pipeline
from typing import Any, Callable

import torch

from noether.data.pipeline import MultiStagePipeline
from noether.data.pipeline.collators import ConcatSparseTensorCollator, DefaultCollator
from noether.data.pipeline.sample_processors import (
    PointSamplingSampleProcessor,
    SupernodeSamplingSampleProcessor,
    ConcatTensorSampleProcessor)
from noether.data import SampleProcessor

from noether.core.schemas.dataset import PipelineConfig

from noether.core.schemas.dataset import ModelDataSpecs

class AnchorPointSamplingSampleProcessor(SampleProcessor):
    """Randomly subsamples points from a pointcloud. Copied from tutorial with some modicications of query/key namings."""

    def __init__(
        self,
        items: set[str],
        num_points: int,
        keep_queries: bool = False,
        seed: int | None = None,
    ):
        """
        Args:
            items: Which pointcloud items should be subsampled (e.g., input_position, output_position, ...). If multiple
              items are present, the subsampling will use identical indices for all items (e.g., to downsample
              output_position and output_pressure with the same subsampling).
            num_points: Number of points to sample.
            seed: Random seed for deterministic sampling for evaluation. Default None (i.e., no seed). If not None,
                requires sample index to be present in batch.
        """
        if not num_points >= 0:
            raise ValueError("Number of points to sample must be non-negative.")

        self.items = items
        self.num_points = num_points
        self.keep_queries = keep_queries
        self.seed = seed

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Subsamples the pointclouds identified by `self.items` with the same subsampling. The outer list and dicts
        are copied explicitly, the Any objects are not. However, the subsampled tensors are "copied" implicitly as
        sampling is implemented via random index access, which implicitly creates a copy of the underlying values.

        Args:
            input_sample: Ssample retrieved from the dataset.

        Returns:
            Preprocessed copy of `input_sample`.
        """

        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)

        # apply preprocessing
        any_item = next(iter(self.items))

        # create perm
        if self.seed is not None:
            if "index" not in output_sample:
                raise ValueError("Sample index is required for deterministic point sampling with a seed.")
            seed = output_sample["index"] + self.seed
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        first_item_tensor = output_sample[any_item]
        assert torch.is_tensor(first_item_tensor)
        if self.keep_queries:
            perm = torch.randperm(len(first_item_tensor), generator=generator)
            if len(first_item_tensor) <= self.num_points:
                discarded_perm = None
            else:
                discarded_perm = perm[self.num_points :]
            perm = perm[: self.num_points]
        else:
            perm = torch.randperm(len(first_item_tensor), generator=generator)[: self.num_points]
            discarded_perm = None
        # subsample
        for item in self.items:
            tensor = output_sample[item]
            output_sample[f"anchor_{item}"] = tensor[perm]
            if discarded_perm is not None:
                output_sample[f"query_{item}"] = tensor[discarded_perm]

        return output_sample


class AtmoPipelineConfig(PipelineConfig):
    seed: int | None = None
    """Random seed for for processes that involve sampling (e.g., point sampling). Defaults to None."""

    num_terrain_points: int
    """Number of terrain points we sample as input for the model. """
    num_obstacles_points: int
    """Number of obstacles points we sample as input for the model. """

    num_terrain_supernodes: int | None = None
    """ Number of terrain supernodes"""
    num_obstacles_supernodes: int | None = None
    """ Number of obstacles supernodes"""
    num_volume_anchor_points: int | None = 0
    """ Number of volume anchor points to sample. Defaults to 0."""

    use_query_positions: bool = False
    """ Whether to also output query positions (in addition to anchors)"""

    data_specs: ModelDataSpecs | None = None

    model_config = {
        "extra": "forbid",}
    

class AtmoPipeline(MultiStagePipeline):
    '''Pipeline for atmospheric simulations
    Applies point subsampling, supernodes selection and anchor selection'''
    def __init__(
        self,
        config:AtmoPipelineConfig
    ):

        seed = config.seed

        terrain_specs = config.data_specs.domains['terrain']
        obstacles_specs = config.data_specs.domains['obstacles']
        volume_specs = config.data_specs.domains['volume']

        # build terrain pipeline
        if len(terrain_specs.feature_dim.keys()) > 0:
            terrain_preproc = [
                 ConcatTensorSampleProcessor(
                    items=terrain_specs.feature_dim.keys(),
                    target_key="terrain_features",
                    dim = -1,
                ),
                PointSamplingSampleProcessor(
                    items={"terrain_position", "terrain_features"},
                    num_points=config.num_terrain_points,
                    seed=None if seed is None else seed + 1,
                ),            
                SupernodeSamplingSampleProcessor(
                    item="terrain_position",
                    num_supernodes=config.num_terrain_supernodes,
                    supernode_idx_key="terrain_supernode_idx",
                    items_at_supernodes={"terrain_features"},
                    seed=None if seed is None else seed + 2,
                ), 
            ]
            terrain_collator = ConcatSparseTensorCollator(
                                items=["terrain_position", "terrain_features"],
                                create_batch_idx=True,
                                batch_idx_key="terrain_batch_idx",
                                )
        else:
            terrain_preproc = [
                PointSamplingSampleProcessor(
                    items={"terrain_position"},
                    num_points=config.num_terrain_points,
                    seed=None if seed is None else seed + 1,
                ),            
                SupernodeSamplingSampleProcessor(
                    item="terrain_position",
                    num_supernodes=config.num_terrain_supernodes,
                    supernode_idx_key="terrain_supernode_idx",
                    seed=None if seed is None else seed + 2,
                ), 
            ]
            terrain_collator = ConcatSparseTensorCollator(
                                items=["terrain_position"],
                                create_batch_idx=True,
                                batch_idx_key="terrain_batch_idx",
                                )

        #build obstacles pipeline
        if len(obstacles_specs.feature_dim.keys()) > 0:
            obstacles_preproc = [
                 ConcatTensorSampleProcessor(
                    items=obstacles_specs.feature_dim.keys(),
                    target_key="obstacles_features",
                    dim = -1,
                ),
                PointSamplingSampleProcessor(
                    items={"obstacles_position", "obstacles_features"},
                    num_points=config.num_obstacles_points,
                    seed=None if seed is None else seed + 3,
                ),            
                SupernodeSamplingSampleProcessor(
                    item="obstacles_position",
                    num_supernodes=config.num_obstacles_supernodes,
                    supernode_idx_key="obstacles_supernode_idx",
                    items_at_supernodes={"obstacles_features"},
                    seed=None if seed is None else seed + 4,
                ), 
            ]
            obstacles_collator = ConcatSparseTensorCollator(
                                items=["obstacles_position", "obstacles_features"],
                                create_batch_idx=True,
                                batch_idx_key="obstacles_batch_idx",
                                )
        else:
            obstacles_preproc = [
                PointSamplingSampleProcessor(
                    items={"obstacles_position"},
                    num_points=config.num_obstacles_points,
                    seed=None if seed is None else seed + 3,
                ),            
                SupernodeSamplingSampleProcessor(
                    item="obstacles_position",
                    num_supernodes=config.num_obstacles_supernodes,
                    supernode_idx_key="obstacles_supernode_idx",
                    seed=None if seed is None else seed + 4,
                ), 
            ]
            obstacles_collator = ConcatSparseTensorCollator(
                                items=["obstacles_position"],
                                create_batch_idx=True,
                                batch_idx_key="obstacles_batch_idx",
                                )

        
        super().__init__(
            sample_processors=[

                *terrain_preproc,
                *obstacles_preproc,

                # sample volume anchors
                AnchorPointSamplingSampleProcessor(
                    items={"volume_position", *volume_specs.output_dims.keys()},
                    num_points=config.num_volume_anchor_points,
                    keep_queries=config.use_query_positions,
                    seed=None if seed is None else seed + 5,
                ),
            ],
            collators=[
                # collate geometry positions (remains sparse for supernode_pooling) #/!\ make features optional
                terrain_collator,
                ConcatSparseTensorCollator(items=["obstacles_supernode_idx"], create_batch_idx=False),
                obstacles_collator,
                ConcatSparseTensorCollator(items=["terrain_supernode_idx"], create_batch_idx=False),
                
                # collate volume data and meteo profile (Dense tensors)
                DefaultCollator(items=[
                    "anchor_volume_position",
                    *[f"anchor_{field}" for field in volume_specs.output_dims.keys()],
                    "meteorological_profile",
                    'index'
                    ],
                    optional_items=[
                        'query_volume_position',
                        *[f"query_{field}" for field in volume_specs.output_dims.keys()]
                    ]),
            ],
        )

    # def __call__(self, samples):
    #     out =  super().__call__(samples)
    #     print({k:v.shape for k,v in out.items()})
    #     return out
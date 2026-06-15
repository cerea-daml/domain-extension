import logging

import torch

from noether.core.schemas.dataset import DatasetSplitIDs
from noether.data import Dataset, with_normalizers
from noether.core.schemas.dataset import StandardDatasetConfig

logger = logging.getLogger(__name__)


class AtmoDataset(Dataset):
    """Dataset implementation for atmospheric datasets with volume fields, terrain and obstacles geometries, and meteorological profiles.
    Modified from by AeroDataset + a bit of CAEMLDataset.
    This class only creates return of very basic fields, it is up to the subclasses to add use-case specific fields
    """

    def __init__(self, dataset_config: StandardDatasetConfig) -> None:
        """

        Args:
            dataset_config: Configuration for the dataset."""
        super().__init__(dataset_config=dataset_config)

        self._raw_dir = dataset_config.root

        self.split = dataset_config.split
        self.design_ids = list(self.get_dataset_splits.model_dump()[self.split])

        logger.info(f"Initialized atmo dataset with {len(self.design_ids)} samples for split '{self.split}'") #FIXME: add specific dataset name

    def __len__(self):
        return len(self.design_ids)
    
    @property
    def get_dataset_splits(self) -> DatasetSplitIDs:
        raise NotImplementedError("Subclasses must implement get_dataset_splits")

    def _load(self, idx: int, varname: str) -> torch.Tensor:
        return self._data[idx][varname]

    @with_normalizers
    def getitem_meteorological_profile(self, idx: int) -> torch.Tensor:
        """Retrieves meteorological_profile (64, 4)"""
        return self._load(idx=idx, varname = 'profile')  # type: ignore[arg-type]
    
    @with_normalizers('positions')
    def getitem_terrain_position(self, idx: int) -> torch.Tensor:
        """Retrieves terrain positions (num_terrain_points, ndim)"""
        return self._load(idx=idx, varname = 'terrain_position')  # type: ignore[arg-type]
    
    @with_normalizers
    def getitem_terrain_rugosity(self, idx: int) -> torch.Tensor:
        """Retrieves terrain rugosity (num_terrain_points, 1)"""
        return self._load(idx=idx, varname = 'terrain_rugosity')  # type: ignore[arg-type]
    
    @with_normalizers('positions')
    def getitem_obstacles_position(self, idx: int) -> torch.Tensor:
        """Retrieves obstacles positions (num_obstacles_points, ndim)"""
        return self._load(idx=idx, varname = 'obstacles_position')  # type: ignore[arg-type]

    @with_normalizers('positions')
    def getitem_volume_position(self, idx: int) -> torch.Tensor:
        """Retrieves volume position (num_volume_points, ndim)"""
        return self._load(idx=idx, varname='volume_position')  # type: ignore[arg-type]

    @with_normalizers
    def getitem_volume_velocity(self, idx: int) -> torch.Tensor:
        """Retrieves volume velocity (num_volume_points, n) where n is either ndim (full field) or 1 (norm)"""
        return self._load(idx=idx, varname='volume_velocity')  # type: ignore[arg-type]
    
    @with_normalizers
    def getitem_volume_tke(self, idx: int) -> torch.Tensor:
        """Retrieves volume turbulent kinetic energy (or k) (num_volume_points, 1)"""
        return self._load(idx=idx, varname='volume_k')  # type: ignore[arg-type]
    
    def getitem_design_id(self, idx: int) -> torch.Tensor:
        '''Design idx for tracking purposes'''
        return torch.tensor([self.design_ids[idx]])
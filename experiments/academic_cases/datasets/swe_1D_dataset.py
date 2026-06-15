from typing import Literal
import logging

import os, re

import numpy as np

import torch

from noether.data import Dataset, with_normalizers
from noether.core.schemas.dataset import StandardDatasetConfig

logger = logging.getLogger(__name__)

class DatasetConfig(StandardDatasetConfig):
    """Base config for datasets with fixed splits."""

    root: str
    """Root directory of the dataset."""
    split: Literal["train", "val", "test", 'test_s2', 'test_s3', 'test_s7', 'test_s10']
    """Which split of the dataset to use."""

class SWE1DDataset(Dataset):
    """Shallow watter in 1D.
    """

    def __init__(self, dataset_config: DatasetConfig) -> None:
        """

        Args:
            dataset_config: Configuration for the dataset. See :class:`~noether.core.schemas.dataset.StandardDatasetConfig` for available options."""
        super().__init__(dataset_config=dataset_config)

        self._root = dataset_config.root
        self.dataset_name = 'swe_1D'

        self.split = dataset_config.split

        self._scale = 1 if self.split in ['train', 'val', 'test'] else int(re.search(r'\d+', self.split).group())

        #data loading and preprocessing
        data = self._load_raw_data()
        idx = self._get_split_idx(dataset_config.split)
        self._position = torch.tensor(data.pop('x')).unsqueeze(-1).to(torch.float32)
        data['h'] = torch.tensor(data['h'])[idx].unsqueeze(-1).to(torch.float32)
        data['u'] = torch.tensor(data['u'])[idx].unsqueeze(-1).to(torch.float32)
        self._temporal_data = data
        self.nt = self._temporal_data['h'].shape[1]

        self._bc = torch.zeros_like(self._position)
        self._bc[0] = self._bc[-1] = 1

        logger.info(f'Initialized dataset {self.dataset_name}, split: {dataset_config.split} with {len(self)} samples')


    def _get_split_idx(self, split):

        assert split in ['train', 'test', 'val', 'test_s2', 'test_s3', 'test_s7', 'test_s10']

        if split == 'train':
            return torch.arange(0, 800)
        if split == 'val':
            return torch.arange(800, 900)
        if split == 'test':
            return torch.arange(900, 1000)
        
        #all large scale splits
        return torch.arange(50)
        

    def __len__(self):
        return self._temporal_data['h'].shape[0] * (self.nt - 1)

    def _load_raw_data(self):
        """Load raw data."""
        return {'x':np.load(self._root + os.sep + f's{self._scale}_pos.npy'),
                'h':np.load(self._root + os.sep + f's{self._scale}_h.npy'),
                'u':np.load(self._root + os.sep + f's{self._scale}_u.npy')}
    
    def _split_idx(self, idx):
        return idx // (self.nt - 1), idx % (self.nt - 1)

    @with_normalizers
    def getitem_position(self, idx: int) -> torch.Tensor:
        """Retrieves positions"""
        return self._position
    
    @with_normalizers
    def getitem_u(self, idx: int) -> torch.Tensor:
        """Retrieves u at time t"""
        idesign, it = self._split_idx(idx)
        return self._temporal_data['u'][idesign, it]
    
    @with_normalizers
    def getitem_h(self, idx: int) -> torch.Tensor:
        """Retrieves h at time t"""
        idesign, it = self._split_idx(idx)
        return self._temporal_data['h'][idesign, it]
    
    @with_normalizers
    def getitem_du(self, idx: int) -> torch.Tensor:
        """Retrieves u(t+1) - u(t))"""
        idesign, it = self._split_idx(idx)
        return self._temporal_data['u'][idesign, it+1] - self._temporal_data['u'][idesign, it]
    
    @with_normalizers
    def getitem_dh(self, idx: int) -> torch.Tensor:
        """Retrieves h(t+1) - h(t))"""
        idesign, it = self._split_idx(idx)
        return self._temporal_data['h'][idesign, it+1] - self._temporal_data['h'][idesign, it]
    
    def getitem_boundary_condition(self, idx: int) -> torch.Tensor:
        '''0/1 indicator of boundary conditions'''
        return self._bc
    
    def getitem_indexes(self, idx:int) -> torch.Tensor:
        '''Retrieve the simulation idx for callbacks'''
        didx, tidx = self._split_idx(idx)
        return torch.tensor([didx, tidx])
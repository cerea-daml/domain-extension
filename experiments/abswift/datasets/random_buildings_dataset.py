#!/usr/bin/env python

'''Torch dataset and pipeline from the random buildings dataset - noether adaptation'''

import os
import json
from pathlib import Path

import numpy as np
import torch

from typing import Literal

from noether.core.schemas.dataset import DatasetSplitIDs
from noether.core.schemas.dataset import StandardDatasetConfig
from noether.data import with_normalizers

from .vtk_tools import *
from .mo_profiles import compute_meteo_profile
from .atmo_dataset import AtmoDataset

class RBDDefaultSplitIDs(DatasetSplitIDs):
    EXPECTED_TRAIN_SIZE = 138
    EXPECTED_TEST_SIZE = 80
    EXPECTED_NEUTRAL_TEST_SUBSET_SIZE: int = 11
    EXPECTED_VAL_SIZE = 10
    EXPECTED_LARGE_SCALE_TEST_SIZE: int = 10
    EXPECTED_VERY_LARGE_SCALE_TEST_SIZE: int = 10
    train: set[int] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227}
    val: set[int] = {80, 81, 82, 83, 84, 85, 86, 87, 88, 89}
    test: set[int] = {90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169}
    neutral_test_subset: set[int]= {96, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139}
    large_scale_test: set[int] = {10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009}
    very_large_scale_test: set[int] = {20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009}

    def _check_no_overlaps(self):
        """Check that splits don't have overlapping IDs. Modified to exclude neutral_test from overlap checking"""
        # Get all split fields (including any additional ones like hidden_test)
        split_fields = {}
        for field_name in self.__class__.model_fields.keys():
            field_value = getattr(self, field_name)
            if isinstance(field_value, set) and field_value:  # Only check non-empty sets
                split_fields[field_name] = field_value

        # Check all pairs of splits for overlaps. Exclude train_subset from this check.
        field_names = [field_name for field_name in split_fields.keys() if field_name not in ["train_subset", "neutral_test_subset"]]
        for i, field1 in enumerate(field_names):
            for field2 in field_names[i + 1 :]:
                overlap = split_fields[field1] & split_fields[field2]
                if overlap:
                    raise ValueError(
                        f"{field1.capitalize()} and {field2} splits have overlapping IDs: {sorted(overlap)}"
                    )
        # Check that train_subset is a subset of training set
        if self.train_subset:
            assert self.train_subset.issubset(self.train), "train_subset is not a subset of the training set"
        # Check that neutral_test_subset is a subset of test set
        if self.neutral_test_subset:
            assert self.neutral_test_subset.issubset(self.test), "neutral_test_subset is not a subset of the test set"

class RandomBuildingsDatasetConfig(StandardDatasetConfig):
    '''Adds support for large and very large splits'''
    split: Literal["train", "val", "test", "neutral_test_subset", "large_scale_test", "very_large_scale_test"]
    """Which split of the dataset to use. Must be one of "train", "val", "test" or "large_scale_test"."""

class RandomBuildingsDataset(AtmoDataset):
    '''
    Dataset of wind flow around randomly placed buildings'''

    STATS_FILE: str = str(Path(__file__).parent / "rbd_statistics.yaml")

    def __init__(self, 
                 config: RandomBuildingsDatasetConfig, 
                 **kwargs):

        self._volume_fields = ['velocity', 'pottemp', 'pressure', 'k', 'epsilon', 'node_type']
        self._invlmos, self._z0s = None, None

        super().__init__(dataset_config=config)

        #load everything in memory
        self._data = dict()
        for idx in range(len(self)):
            self._data[idx] = self._load_and_preprocess(idx)

    @property
    def get_dataset_splits(self) -> DatasetSplitIDs:
        return RBDDefaultSplitIDs()
    
    def _load_invlmos_z0s(self):
        #parse info for lmo, z0 (must be changed later I think)
        with open(self._raw_dir+os.sep+'random_buildings_dataset.json', 'r') as f:  #This might need to be changed to pure config driven stuff
            info = json.load(f)
            self._invlmos = info['invlmo']
            self._z0s = info['z0']
            self._large_scale_z0s = info['ls_z0s']

    def _load_and_preprocess(self, idx: int):
        '''Loads data in-memory, and runs necessary preprocessing'''

        file_id = self.design_ids[idx]
        if 10000 <= file_id < 20000:
            file_id = f'ls_{file_id%10000}'
        elif 20000 <= file_id < 30000:
            file_id = f'vls_{file_id%20000}'

        data = {}

        ####### PROCESS PROFILE #########
        if self._invlmos is None: #invlmos and z0s haven't been loaded yet
            self._load_invlmos_z0s()

        if self.split == "large_scale_test" or self.split == "very_large_scale_test": #same params were chosen for both
            invlmo = 0.0
            z0 = self._large_scale_z0s[idx]
        else:
            invlmo = self._invlmos[file_id]
            z0 = self._z0s[file_id]

        lmo = 1 / invlmo if invlmo !=0 else 1e50
        z = np.linspace(0, 100, 64)
        profile = compute_meteo_profile(lmo = lmo, z0 = z0, zref=80.0, uref=6.0, t0=293.15, z = z)
        data['profile'] = profile

        ####### PROCESS VOLUME #######
        vtk_volume = read_vtk(f'{self._raw_dir}/volume_{file_id}.vtk')
        data['volume_position'] = get_coords(vtk_volume, loc = 'cells')

        #volumic fields
        fields = {f'volume_{f}':get_field(vtk_volume, f, loc = 'cells') for f in self._volume_fields}
        fields = {k:v[:,None] if len(v.shape)==1 else v  for k,v in fields.items()} #add extra dim for scalar fields
        data = {**data, **fields}

        ########### PROCESS BUILDINGS ##########
        vtk_buildings = read_vtk(f'{self._raw_dir}/buildings_{file_id}.vtk')
        data['obstacles_position'] = get_coords(vtk_buildings, loc = 'points')

        ########## PROCESS GROUND ############
        vtk_ground = read_vtk(f'{self._raw_dir}/ground_{file_id}.vtk')
        data['terrain_position'] = get_coords(vtk_ground, loc = 'points')

        #rugosity
        n_pts = data['terrain_position'].shape[0]
        data['terrain_rugosity'] = np.full((n_pts, 1), z0)

        #invlmo on terrain, since we have it in paper
        data['terrain_invlmo'] = np.full((n_pts, 1), invlmo)

        #tensorify
        data = {k:torch.tensor(v, dtype=torch.float32) for k,v in data.items()}

        return data
    
    def get_volume_mesh(self, idx):
        '''Return the vtk volume mesh (for plots)'''
        file_id = self.design_ids[idx]
        if 10000 <= file_id < 20000:
            file_id = f'ls_{file_id%10000}'
        elif 20000 <= file_id < 30000:
            file_id = f'vls_{file_id%20000}'
        vtk_volume = read_vtk(f'{self._raw_dir}/volume_{file_id}.vtk')
        return vtk_volume
    

    def area_mask(self, coords: torch.Tensor):

        device = coords.device

        coords = self.normalizers['positions'].inverse(coords.cpu())

        coords = coords[...,:2]

        if self.split == "very_large_scale_test":
            scaling_factor = 5
        elif self.split == "large_scale_test":
            scaling_factor = 2
        else:
            scaling_factor = 1

        cmin = torch.tensor([-50, -50]) * scaling_factor
        cmax = torch.tensor( [50, 50] ) * scaling_factor

        bat_id = (coords > cmin) & (coords < cmax)

        bat_id = torch.all(bat_id, axis = - 1).to(device)

        return {
            'buildings': bat_id,
            'volume': ~bat_id
        }
    
    @with_normalizers
    def getitem_terrain_invlmo(self, idx: int) -> torch.Tensor:
        """Retrieves 1 / Lmo for all terrain points (num_terrain_points, 1)"""
        return self._load(idx=idx, varname = 'terrain_invlmo')  # type: ignore[arg-type]
    
    @with_normalizers
    def getitem_volume_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves volume pressure (num_volume_points, 1)"""
        return self._load(idx=idx, varname='volume_pressure')  # type: ignore[arg-type]
    
    @with_normalizers
    def getitem_volume_pottemp(self, idx: int) -> torch.Tensor:
        """Retrieves volume potential_temperature (num_volume_points, 1)"""
        return self._load(idx=idx, varname='volume_pottemp')  # type: ignore[arg-type]
    
    @with_normalizers
    def getitem_volume_epsilon(self, idx: int) -> torch.Tensor:
        """Retrieves volume turbulent kinnetic energy dissipation rate (or epsilon) (num_volume_points, 1)"""
        return self._load(idx=idx, varname='volume_epsilon')  # type: ignore[arg-type]
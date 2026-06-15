from __future__ import annotations

import os
import numpy as np

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

from typing import Literal

from pydantic import Field
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig

from noether.data.schemas import ModelDataSpecs

from abswift.datasets.vtk_tools import get_slice
from .plot import plot

class PlotCallbackConfig(PeriodicDataIteratorCallbackConfig):
    name: Literal["PlotCallback"] = Field("PlotCallback", frozen=True)

    plot_height: int
    save_folder: str = 'figures'

    data_specs: ModelDataSpecs

    save_res_array: bool = False
    """whether to save the predicted field arrays (using np.save) or not. Only saves the velocity currently"""

class PlotCasesCallback(PeriodicDataIteratorCallback):
    """A periodic Callback that is invoked at the end of each epoch to generate prediction plots."""

    def __init__(self, callback_config: PlotCallbackConfig, **kwargs):
        """

        Args:
            callback_config: configuration of the PlotCallback. See :class:`~abswift.schemas.callbacks.PlotCallbackConfig` for the available options.
        """

        super().__init__(callback_config=callback_config, **kwargs)
        self.dataset_key = callback_config.dataset_key
        self.plot_height = callback_config.plot_height

        # kinda hacky but couldn't figure out where else to get it
        self.save_path = self.writer.path_provider.run_output_path / callback_config.save_folder
        self.save_path.mkdir(exist_ok=True)

        self._dataset = self.data_container.get_dataset(self.dataset_key)

        self.data_specs = callback_config.data_specs
        self.volume_fields = self.data_specs.domains['volume'].output_dims.keys()

        self.save_res_array = callback_config.save_res_array

        os.makedirs(self.save_path, exist_ok=True)

    def _process_one_case(self, batch, *, trainer_model, **_):

        idx = batch['index'][0].item()
        design_id = self._dataset.getitem_design_id(idx)

        mesh = self._dataset.get_volume_mesh(idx)
        triang, ids = get_slice(mesh, self.plot_height)

        #setup queries
        volume_pos = self._dataset.getitem_volume_position(idx)
        batch['query_volume_position'] = volume_pos[ids].unsqueeze(0).to(self.model.device)

        #run inference
        forward_inputs = {k: v for k, v in batch.items() if k in self.trainer.forward_properties}
        with self.trainer.autocast_context:
            model_prediction = self.model(**forward_inputs)

        # extract data
        preds = {f:model_prediction[f'query_{f}'][0] for f in self.data_specs.volume_output_dims.keys()}
        data = self._dataset[idx]
        labels = {f:data[f][ids] for f in self.data_specs.volume_output_dims.keys()}

        #to cpu
        preds = {k:v.cpu() for k,v in preds.items()}
        labels = {k:v.cpu() for k,v in labels.items()}

        #denormalise
        preds = {k:self._dataset.normalizers[k].inverse(v) for k,v in preds.items()}
        labels = {k:self._dataset.normalizers[k].inverse(v) for k,v in labels.items()}

        #numpyfy
        preds = {k:v.numpy() for k,v in preds.items()}
        labels = {k:v.numpy() for k,v in labels.items()}

        #generate plot
        fig = plot(triang, labels, preds)
        fig.savefig(self.save_path / f'{self.dataset_key}_{design_id}')
        
        self.logger.info(f'Generated plot {self.dataset_key}_{design_id}')
        if self.save_res_array and design_id!=9:
            np.save(self.save_path / f'{self.dataset_key}_{design_id}.npy', preds['volume_velocity'])

    def process_data(self, batch, *, trainer_model):
        
        assert 'index' in batch.keys(), 'indexes not provided but they are requiered to plot'

        assert len(batch['index']) ==  1, f'Batch size > 1 not implemented, got batch_size={len(batch["index"])}'

        self._process_one_case(batch, trainer_model=trainer_model)
# modified from https://github.com/Emmi-AI/noether/blob/main/tutorial/callbacks/surface_volume_evaluation_metrics.py

import os

import torch

from typing import Literal
from pydantic import Field

from noether.core.distributed import all_gather_nograd, all_gather_nograd_clipped
from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig

from academic_cases.datasets import AcademicDataSpecs

import numpy as np

class RolloutMetricsCallbackConfig(PeriodicDataIteratorCallbackConfig):
    name: Literal["RolloutMetricsCallback"] = "RolloutMetricsCallback"

    forward_properties: list[str] = []
    """List of properties in the dataset to be forwarded during inference."""
    batch_size: int = Field(1)
    """Batch size for evaluation. Currently only batch_size=1 is supported."""
    data_specs: AcademicDataSpecs
    """Data specs, used to split piped data and prediction into separate fields"""
    times: list[int]
    """Time horizons on which to compute metrics"""
    save_folder: None | str = None

class RolloutMetricsCallback(PeriodicDataIteratorCallback):
    """ Computes metrics over denormalised fields with a rollout from initial time step.
    """

    def __init__(self, callback_config: RolloutMetricsCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

        self._config = callback_config
        self.dataset_key = callback_config.dataset_key
        self.normalizers = self.data_container.get_dataset(self.dataset_key).normalizers
        self.forward_properties = callback_config.forward_properties
        self.data_specs = callback_config.data_specs
        self.times = callback_config.times
        self.data_nt = self.data_container.get_dataset(self.dataset_key).nt
        self.Nt = max(callback_config.times)

        self._dataset = self.data_container.get_dataset(self.dataset_key)

        assert self.Nt < self.data_nt, f'rollout length ({self.Nt}) exeeds dataset temporal length ({self.data_nt})'

        if callback_config.save_folder is not None:
            self.save_path = self.writer.path_provider.run_output_path / callback_config.save_folder
            os.makedirs(self.save_path, exist_ok=True)
        else:
            self.save_path = None

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        """
        Execute forward pass and compute metrics.

        Args:
            batch: Input batch dictionary
            **_: Additional unused arguments

        Returns:
            Dictionary mapping metric names to computed values
        """

        assert 'index' in batch.keys(), 'index must be provided'
        assert len(batch['index']) == 1, 'Batch size > 1 not supported'

        idx = batch['index'].item()
        didx, tidx = self._dataset.getitem_indexes(idx).squeeze()
        didx, tidx = didx.item(), tidx.item()
        if tidx != 0:
            return

        inputs = {k:batch.get(k) for k in self.trainer.forward_properties}
        with self.trainer.autocast_context:
            prediction_history = self._rollout(inputs)

        gt_history = { #FIXME: not very robust
            k: self._dataset._temporal_data[k][didx, :self.Nt+1].squeeze()
                for k in self.data_specs.input_features.keys() if k != "boundary_condition"
        }


        if self.save_path is not None:
            np.save(self.save_path / f'gt_{self.dataset_key}_{idx}.npy', gt_history['V'])
            np.save(self.save_path / f'pred_{self.dataset_key}_{idx}.npy', prediction_history['V'])

        metrics = {}
        for field in self.data_specs.input_features.keys():
            if field != 'boundary_condition':
                for it in self.times:
                    metrics.update(self._compute_metrics(prediction_history[field][it], gt_history[field][it], f'it{it}_{field}'))

        return metrics
    
    @torch.no_grad()
    def _rollout(self, inputs):
        inputs = inputs.copy()
        input_specs = self.data_specs.input_features
        output_specs = self.data_specs.output_targets

        hist = dict(zip(input_specs.keys(), inputs['input_feature'].cpu().split(list(input_specs.values()), dim = -1)))
        
        hist = {k:[v] for k,v in hist.items()}

        for it in range(self.Nt):
            pred = self.model(**inputs)['prediction']
            
            # denormalise, integrate, renormalise
            delta = dict(zip(output_specs.keys(), pred.cpu().split(list(output_specs.values()), dim = -1)))
            delta = {k:self.normalizers[k].inverse(v) for k,v in delta.items()}
            
            fields = {k: self.normalizers[k].inverse(v[-1]) if k in self.normalizers.keys() else v[-1] for k,v in hist.items()}
            fields = {k: fields[k] + delta.get('d'+k, 0) for k in fields.keys()}
            fields = {k: self.normalizers[k](v) if k in self.normalizers.keys() else v
                            for k,v in fields.items()}

            inputs['input_feature'] = torch.concatenate(list(fields.values()), axis = -1).to(self.model.device)

            {k:v.append(fields[k].cpu()) for k,v in hist.items()}

        #stack, denormalise and return
        if 'boundary_condition' in hist.keys(): hist.pop('boundary_condition')
        hist = {k:torch.stack(v, axis = 0).squeeze() for k,v in hist.items()}
        hist = {k:self.normalizers[k].inverse(v) for k,v in hist.items()}
        return hist

    def _compute_metrics(
        self, denormalized_predictions: torch.Tensor, denormalized_targets: torch.Tensor, field_name: str
    ) -> dict[str, torch.Tensor]:
        """
        Compute evaluation metrics for predictions vs targets.

        Calculates Mean Squared Error (MSE), Mean Absolute Error (MAE),
        and relative L2 error for the given field.

        Args:
            denormalized_predictions: Denormalized prediction tensor
            denormalized_targets: Denormalized target tensor
            field_name: Name of the field being evaluated (used for metric naming)

        Returns:
            Dictionary mapping metric names to computed values
        """
        delta = denormalized_predictions - denormalized_targets

        metrics = {
            f"{field_name}_L1error": torch.linalg.norm(delta, ord = 1) / torch.linalg.norm(denormalized_targets, ord = 1),
            f"{field_name}_L2error": torch.linalg.norm(delta, ord = 2) / torch.linalg.norm(denormalized_targets, ord = 2),
            f"{field_name}_NMSE": torch.linalg.norm(delta, ord = 2) / torch.var(denormalized_targets),
        }

        return metrics

    def process_results(self, results: dict[str, torch.Tensor], **_) -> None:
        """
        Log computed metrics to writer.

        Args:
            results: Dictionary of computed metrics
            **_: Additional unused arguments
        """
        if not results:
            self.logger.warning(f"No metrics computed for dataset '{self.dataset_key}'")
            return

        for name, metric in results.items():
            metric_key = f"loss/{self.dataset_key}/{name}"
            self.writer.add_scalar(
                key=metric_key,
                value=metric.mean(),
                logger=self.logger,
                format_str=".6f",
            )

        self.logger.debug(f"Logged {len(results)} metrics for dataset '{self.dataset_key}'")


    @staticmethod
    def _collate_result(result, global_dataset_len):
        '''Overriden to remove None from results. FIXME: will still fail if first item is None...'''
        if isinstance(result[0], dict):
            # tuple[dict] -> dict[tensor]
            result = {
                k: PeriodicDataIteratorCallback._collate_tensors([r[k] for r in result if r is not None]) for k in result[0].keys()
            }
            result = {k: all_gather_nograd_clipped(v, global_dataset_len) for k, v in result.items()}
        else:
            if isinstance(result[0], list):
                # List[List[Tensor]] -> List[Tensor]
                result = [torch.concat(item) for item in zip(*result, strict=True)]
                result = [all_gather_nograd_clipped(item, global_dataset_len) for item in result]
            elif result[0] is None:
                return None
            else:
                if torch.is_tensor(result[0]):
                    # List[Tensor] -> Tensor
                    if result[0].ndim == 0:
                        result = torch.stack(result)
                    else:
                        result = torch.concat(result)
                else:
                    result = torch.tensor(result)
                result = all_gather_nograd_clipped(result, global_dataset_len)
        return result

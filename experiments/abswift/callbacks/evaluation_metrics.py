# Modified from https://github.com/Emmi-AI/noether/blob/main/tutorial/callbacks/surface_volume_evaluation_metrics.py

import torch

from typing import Literal

from pydantic import Field

from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig
from noether.core.callbacks.periodic import PeriodicDataIteratorCallback

from noether.data.schemas import ModelDataSpecs

class EvaluationMetricsCallbackConfig(PeriodicDataIteratorCallbackConfig):
    name: Literal["EvaluationMetricsCallback"] = "EvaluationMetricsCallback"

    # forward_properties: list[str] = []
    # """List of properties in the dataset to be forwarded during inference."""
    batch_size: int = Field(1)
    """Batch size for evaluation. Currently only batch_size=1 is supported."""
    data_specs: ModelDataSpecs
    """Data specs, used to split piped data and prediction into separate fields"""
    metrics_per_area: bool
    """Enable per-area metrics computations. Areas should be defined with area_mask method in the dataset """

class EvaluationMetricsCallback(PeriodicDataIteratorCallback):
    """ Computes metrics over denormalised fields.
    """

    def __init__(self, callback_config: EvaluationMetricsCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

        self._config = callback_config
        self.dataset_key = callback_config.dataset_key
        self.dataset_normalizers = self.data_container.get_dataset(self.dataset_key).normalizers
        self.data_specs = callback_config.data_specs

        self.metrics_per_area = callback_config.metrics_per_area

        self.volume_fields = self.data_specs.domains['volume'].output_dims.keys()

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        """
        Execute forward pass and compute metrics.

        Args:
            batch: Input batch dictionary
            **_: Additional unused arguments

        Returns:
            Dictionary mapping metric names to computed values
        """

        #run inference
        forward_inputs = {k: v for k, v in batch.items() if k in self.trainer.forward_properties}
        with self.trainer.autocast_context:
            model_prediction = self.model(**forward_inputs)

        # extract data
        preds = {f:model_prediction[f'query_{f}'] for f in self.volume_fields}
        labels = {f:batch[f'query_{f}'] for f in self.volume_fields}

        #to cpu
        preds = {k:v.cpu() for k,v in preds.items()}
        labels = {k:v.cpu() for k,v in labels.items()}

        #denormalise
        preds = {k:self.dataset_normalizers[k].inverse(v) for k,v in preds.items()}
        labels = {k:self.dataset_normalizers[k].inverse(v) for k,v in labels.items()}

        metrics = {}
        for field in self.volume_fields:
            y_true, y_pred = labels[field], preds[field]
            metrics[f'{field}_l1_err'] = torch.linalg.norm(y_true - y_pred, ord = 1, axis = (1, 2)) / torch.linalg.norm(y_true, ord = 1, axis = (1, 2))
            metrics[f'{field}_l2_err'] = torch.linalg.norm(y_true - y_pred, ord = 2, axis = (1, 2)) / torch.linalg.norm(y_true, ord = 2, axis = (1, 2))

            metrics[f'{field}_nmse'] = torch.mean(torch.square(y_true - y_pred), axis = (1, 2)) / torch.var(y_true, axis = (1, 2))

        if self.metrics_per_area:
            assert batch['query_volume_position'].shape[0] == 1, 'Batch size > 1 not supported for area metrics'
            coords = batch['query_volume_position'].squeeze().cpu()
            areas = self.data_container.get_dataset(self.dataset_key).area_mask(coords)
            for field in self.volume_fields:
                y_true, y_pred = labels[field], preds[field]
                for area, mask in areas.items():
                    area_y_true = y_true[0, mask, :].unsqueeze(0)
                    area_y_pred = y_pred[0, mask, :].unsqueeze(0)
                    metrics[f'{area}_{field}_l1_err'] = torch.linalg.norm(area_y_true - area_y_pred, ord = 1, axis = (1, 2)) / torch.linalg.norm(area_y_true, ord = 1, axis = (1, 2))
                    metrics[f'{area}_{field}_l2_err'] = torch.linalg.norm(area_y_true - area_y_pred, ord = 2, axis = (1, 2)) / torch.linalg.norm(area_y_true, ord = 2, axis = (1, 2))

                    metrics[f'{area}_{field}_nmse'] = torch.mean(torch.square(area_y_true - area_y_pred), axis = (1, 2)) / torch.var(area_y_true, axis = (1, 2))

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
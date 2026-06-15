# modified from https://github.com/Emmi-AI/noether/blob/main/tutorial/callbacks/surface_volume_evaluation_metrics.py

import torch

from typing import Literal
from pydantic import Field

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig

from academic_cases.datasets import AcademicDataSpecs

class EvaluationMetricsCallbackConfig(PeriodicDataIteratorCallbackConfig):
    name: Literal["EvaluationMetricsCallback"] = "EvaluationMetricsCallback"

    forward_properties: list[str] = []
    """List of properties in the dataset to be forwarded during inference."""
    batch_size: int = Field(1)
    """Batch size for evaluation. Currently only batch_size=1 is supported."""
    data_specs: AcademicDataSpecs
    """Data specs, used to split piped data and prediction into separate fields"""

class EvaluationMetricsCallback(PeriodicDataIteratorCallback):
    """ Computes metrics over denormalised fields.
    If on_query is set to true (default), will ony compute metrics from "query" points
    """

    def __init__(self, callback_config: EvaluationMetricsCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

        self._config = callback_config
        self.dataset_key = callback_config.dataset_key
        self.dataset_normalizers = self.data_container.get_dataset(self.dataset_key).normalizers
        self.forward_properties = callback_config.forward_properties
        self.data_specs = callback_config.data_specs

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        """
        Execute forward pass and compute metrics.

        Args:
            batch: Input batch dictionary
            **_: Additional unused arguments

        Returns:
            Dictionary mapping metric names to computed values
        """
        model_outputs = self._run_model_inference(batch)

        predicions, targets = self._process_model_output_and_data(model_outputs['prediction'], batch['target_feature'])

        metrics = {}
        for field in self.data_specs.output_targets.keys():
            metrics.update(self._compute_metrics(predicions[field], targets[field], field))

        return metrics
    
    def _process_model_output_and_data(self, prediction, target):
        '''Split the prediction / target into fields and denormalise'''

        i = 0
        predictions, targets = {}, {}
        for field, fdim in self.data_specs.output_targets.items():
            f_pred = prediction[..., i:i+fdim]
            f_target = target[..., i:i+fdim]
            i+=fdim

            f_pred, f_target = self._denormalize(f_pred, f_target, field)
            predictions[field] = f_pred
            targets[field] = f_target

        return predictions, targets

    def _run_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Run model inference, optionally in chunks.

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary of model outputs
        """
        forward_inputs = {k: v for k, v in batch.items() if k in self.forward_properties}
        with self.trainer.autocast_context:
            return self.model(**forward_inputs)

    def _denormalize(
        self, predictions: torch.Tensor, targets: torch.Tensor, key:str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Denormalize predictions and targets using the appropriate normalizer.

        This method finds the specific normalizer for the given key and uses it to denormalize,
        instead of calling pipeline.denormalize which would process the entire pipeline.

        Args:
            predictions: Tensor containing the predictions to denormalize
            targets: Tensor containing the targets to denormalize
            key: Key to identify the normalizer for denormalization

        Returns:
            Tuple of (denormalized_predictions, denormalized_targets)

        Raises:
            KeyError: If no normalizer is found for the given key
        """
        try:
            normalizer = self.dataset_normalizers[key]
        except KeyError as e:
            raise KeyError(
                f"No normalizer found for key '{key}'. Available normalizers: {list(self.dataset_normalizers.keys())}"
            ) from e

        denormalized_predictions = normalizer.inverse(predictions.cpu())
        denormalized_targets = normalizer.inverse(targets.cpu())
        return denormalized_predictions, denormalized_targets

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
        assert denormalized_predictions.shape[0] == denormalized_targets.shape[0] == 1, f'batch size > 1 not supported, got {denormalized_predictions.shape[0]}'
        denormalized_targets = denormalized_targets.squeeze()
        denormalized_predictions = denormalized_predictions.squeeze()
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
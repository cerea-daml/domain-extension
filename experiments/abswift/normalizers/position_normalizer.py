from typing import Any

import torch


from noether.data.preprocessors import PreProcessor, to_tensor

from noether.data.preprocessors.normalizers import (
    FieldNormalizerConfig, 
    MeanStdNormalization, 
    MeanStdNormalizerConfig,
    PositionNormalizerConfig,
    ShiftAndScaleNormalizer,
    ShiftAndScaleNormalizerConfig,
)

class PositionNormalizer2(ShiftAndScaleNormalizer):
    """Normalizes position data by scale / (raw_pos_max - raw_pos_min) and shift of -raw_pos_min. It inherits from ShiftAndScaleNormalizer. Unlike the original implemetation, this does not check if normalised position are within [0, scale]
    Modified version that does not verify if position are within bounds, as they exeed them in large scale cases"""

    def __init__(
        self,
        normalizer_config: PositionNormalizerConfig,
        **kwargs,
    ):
        """

        Args:
            normalizer_config: Configuration containing raw position min, max, and scale values. See :class:`~noether.core.schemas.normalizers.PositionNormalizerConfig` for details.
            **kwargs: Additional arguments passed to the parent class.

        Raises:
            ValueError: If `raw_pos_min` and `raw_pos_max` do not have the same length.
            ValueError: If `raw_pos_max` is equal to `raw_pos_min`.
            ValueError: If `scale` is not a positive number.
        """

        assert normalizer_config.zero_center == False, 'Zero center not implemented'

        self.raw_pos_min = normalizer_config.raw_pos_min
        self.raw_pos_max = normalizer_config.raw_pos_max
        # Do not remove this. The scale variable is not the same as we pass to the ShiftAndScaleNormalizer.
        # It is used to scale the coordinates to a range of [0, scale]. However, we need to recompute the scale based on the raw position min and max values.
        scale = to_tensor(normalizer_config.scale)

        self.resizing_scale = scale  # this is a reference to the input scale, not the computed scale

        scale = scale / (self.raw_pos_max - self.raw_pos_min)
        shift = - self.raw_pos_min

        super().__init__(
            normalizer_config=ShiftAndScaleNormalizerConfig(
                shift=shift,
                scale=scale,
            ),
            **kwargs,
        )

    def __call__(self, x: Any) -> Any:
        """Applies the position normalization to the input tensor.

        Args:
            x: torch.Tensor: The input tensor to normalize.

        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        output = super().__call__(x)  # type: ignore[return-value]
        # if torch.any(output < 0) or torch.any(output > self.resizing_scale):
        #     raise ValueError("Normalized positions are out of bounds [0, scale].")

        return output

class FieldNormalizer2(PreProcessor):
    """Preprocessor that normalizes a field based on a specified strategy and dataset statistics.
    Modified to support modified position normalizer"""

    normalizer: PreProcessor

    def __init__(
        self,
        normalizer_config: FieldNormalizerConfig,
        statistics: dict[str, list[float | int] | float | int] | None,
        **kwargs,
    ):
        """

        Args:
            normalizer_config: Configuration containing the normalization strategy and logscale flag. See :class:`~noether.core.schemas.normalizers.FieldNormalizerConfig` for details.
            statistics: A dictionary containing the dataset statistics needed for normalization (e.g., mean, std, raw_pos_min, raw_pos_max).
            **kwargs: Additional arguments passed to the parent class.

        Raises:
            ValueError: If the required statistics for the chosen strategy are not present in the `statistics` dictionary.
            ValueError: If the normalization strategy is not supported.
        """
        super().__init__(**kwargs)

        stat_keys = normalizer_config.stat_keys or {}

        if statistics is None:
            raise ValueError("Statistics must be provided for FieldNormalizer.")

        if normalizer_config.strategy == "mean_std":
            mean_key = stat_keys.get("mean", f"{self.normalization_key}_mean")
            std_key = stat_keys.get("std", f"{self.normalization_key}_std")
            mean_val = statistics[mean_key]
            std_val = statistics[std_key]
            if isinstance(mean_val, (int, float)):
                mean_val = [mean_val]
            if isinstance(std_val, (int, float)):
                std_val = [std_val]
            self.normalizer = MeanStdNormalization(
                MeanStdNormalizerConfig(
                    mean=mean_val,
                    std=std_val,
                    logscale=normalizer_config.logscale,
                ),
                normalization_key=self.normalization_key,
            )
        elif normalizer_config.strategy == "position" or normalizer_config.strategy == "min_max":
            min_key = stat_keys.get("min", f"{self.normalization_key}_min")
            max_key = stat_keys.get("max", f"{self.normalization_key}_max")
            if min_key not in statistics:
                raise ValueError(
                    f"Missing required statistics for position normalization: '{min_key}' and/or '{max_key}' not found in statistics."
                )
            if max_key not in statistics:
                raise ValueError(
                    f"Missing required statistics for position normalization: '{min_key}' and/or '{max_key}' not found in statistics."
                )
            self.normalizer = PositionNormalizer2(
                PositionNormalizerConfig(
                    raw_pos_min=statistics[min_key],
                    raw_pos_max=statistics[max_key],
                    scale=normalizer_config.scale,
                    zero_center=normalizer_config.zero_center,
                ),
                normalization_key=self.normalization_key,
            )
        else:
            raise ValueError(f"Unknown normalizer type '{normalizer_config.strategy}'")

    def __call__(self, x: Any) -> Any:
        return self.normalizer(x)  # type: ignore[return-value]

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denormalize(x)  # type: ignore[return-value]
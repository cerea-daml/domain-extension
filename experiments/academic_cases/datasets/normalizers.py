from typing import Any

import torch

from noether.data.preprocessors.normalizers import ShiftAndScaleNormalizer

from noether.data.preprocessors import to_tensor

from noether.core.schemas.normalizers import (
    PositionNormalizerConfig,
    ShiftAndScaleNormalizerConfig,
)

class PositionNormalizer(ShiftAndScaleNormalizer):
    """Normalizes position data to a range of [-scale/2, scale/2]. It inherits from ShiftAndScaleNormalizer and applies a shift and scale based on the provided raw position min and max values.
    Modified version that centers positions and that does not verify if position are within bounds, as they exeed them in large scale cases"""

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

        self.raw_pos_min = normalizer_config.raw_pos_min
        self.raw_pos_max = normalizer_config.raw_pos_max
        # Do not remove this. The scale variable is not the same as we pass to the ShiftAndScaleNormalizer.
        # It is used to scale the coordinates to a range of [0, scale]. However, we need to recompute the scale based on the raw position min and max values.
        scale = to_tensor(normalizer_config.scale)

        self.resizing_scale = scale  # this is a reference to the input scale, not the computed scale

        scale = scale / (self.raw_pos_max - self.raw_pos_min)
        shift = - self.raw_pos_min   - (self.raw_pos_max - self.raw_pos_min) / 2 #center around 0

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
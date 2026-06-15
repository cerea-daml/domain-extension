# Simple generic trainer

import torch
import torch.nn.functional as F

from noether.training.trainers import BaseTrainer

class Trainer(BaseTrainer):
    """Trainer class. Computes MSE between "prediction" and "target feature", and optionally another term if queries are present"""

    def loss_compute(
            self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            """Given the output of the model and the targets, compute the losses.
            Args:
                forward_output The output of the model, containing the predictions for each output mode.
                targets: Dict containing all target values to compute the loss.

            Returns:
                A dictionary containing the computed losses for each output mode.
            """

            losses = {}

            prediction = forward_output['prediction']
            target = targets['target_feature']

            losses['target'] = F.mse_loss(prediction, target)

            return losses
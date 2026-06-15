# Simple generic trainer that compute loss for each fields in the predicted data

import torch
import torch.nn.functional as F

from noether.training.trainers import BaseTrainer

class MultiLossTrainer(BaseTrainer):
    """Trainer class for loss on multiple fields.
    Simply computes mse over each field"""

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

            targets.pop('index', None)

            assert set(forward_output.keys()) == set(targets.keys()), f"Model outputs and targets keys don't match, fwd output: {forward_output.keys()}, target: {targets.keys()}" #TODO: BaseTrainer has some batch / target keys already registered

            losses: dict[str, torch.Tensor] = {}

            for field in forward_output.keys():
                losses[field] = F.mse_loss(targets[field], forward_output[field])

            if len(losses) == 0:
                raise ValueError("No losses computed, check your output keys and loss function.")

            return losses
"""
Transition-ensemble disagreement penalty.

Maintains *k* independent TransitionEstimators.  For any (s, a) pair the
penalty is the maximum pairwise L2 distance among the k predicted next
states, returned as a [B, 1] tensor ready to be subtracted from the
IQ-Learn reward inside ``iq_loss``.
"""

import torch
import torch.nn.functional as F
from typing import List

from agent.transition import TransitionEstimator


class TransitionEnsemble:
    def __init__(self, k: int, obs_dim: int, action_dim: int, args,
                 device=None):
        self.k = k
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = (torch.device(args.device)
                       if device is None else device)
        self.estimators: List[TransitionEstimator] = [
            TransitionEstimator(obs_dim, action_dim, args, self.device)
            for _ in range(k)
        ]

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def train(self, training: bool = True):
        for est in self.estimators:
            est.train(training)

    def update(self, obs: torch.Tensor, action: torch.Tensor,
               next_obs: torch.Tensor, logger=None, step: int = 0) -> dict:
        """Train every estimator on the *same* batch (obs, action, next_obs).
        `action` should already be float (one-hot for discrete envs).
        """
        all_losses = {}
        for i, est in enumerate(self.estimators):
            ld = est.update(obs, action, next_obs, logger, step)
            for key, val in ld.items():
                all_losses[f"ensemble_{i}/{key}"] = val
        return all_losses

    def update_from_buffer(self, replay_buffer, batch_size: int,
                           logger=None, step: int = 0) -> dict:
        """Each estimator samples its own mini-batch (bootstrap effect)."""
        all_losses = {}
        for i, est in enumerate(self.estimators):
            ld = est.update_from_buffer(replay_buffer, batch_size,
                                        logger, step)
            for key, val in ld.items():
                all_losses[f"ensemble_{i}/{key}"] = val
        return all_losses

    # ------------------------------------------------------------------ #
    #  Penalty computation                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _action_to_onehot(action: torch.Tensor,
                          action_dim: int) -> torch.Tensor:
        """[B, 1] int  ->  [B, action_dim] float one-hot."""
        return F.one_hot(
            action.long().squeeze(-1), action_dim
        ).float()

    @torch.no_grad()
    def penalty(self, obs: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """Max pairwise L2 distance among k predicted next-states.

        Args:
            obs:    [B, obs_dim]
            action: [B, 1] integer  (will be one-hot encoded internally)
        Returns:
            [B, 1]  non-negative penalty.
        """
        action_oh = self._action_to_onehot(action, self.action_dim)

        # [k, B, obs_dim]
        preds = torch.stack([
            est.model.mean(obs, action_oh)
            for est in self.estimators
        ], dim=0)

        # Max pairwise L2 over the k predictions.
        # For moderate k (<=10) the O(k^2) loop is cheap.
        max_dist = torch.zeros(obs.shape[0], 1, device=obs.device)
        for i in range(self.k):
            for j in range(i + 1, self.k):
                dist = torch.norm(
                    preds[i] - preds[j], p=2, dim=-1, keepdim=True)
                max_dist = torch.maximum(max_dist, dist)

        return max_dist

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str, suffix: str = ""):
        for i, est in enumerate(self.estimators):
            est.save(path, suffix=f"{suffix}_ensemble_{i}")

    def load(self, path: str, suffix: str = ""):
        for i, est in enumerate(self.estimators):
            est.load(path, suffix=f"{suffix}_ensemble_{i}")

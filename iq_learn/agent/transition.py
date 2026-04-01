"""
Transition distribution estimator T(s'|s, a).
Models the environment dynamics as a diagonal Gaussian:
    p(s' | s, a) = N(μ_θ(s, a), diag(σ_θ(s, a)²))
The network is trained by maximising the log-likelihood of (s, a, s')
tuples drawn directly from an expert (or replay) dataset, so no
additional environment interaction is required.
Usage
-----
    estimator = TransitionEstimator(obs_dim, action_dim, args, device)
    loss_info  = estimator.update(obs, action, next_obs, logger, step)
    log_p      = estimator.log_prob(obs, action, next_obs)   # [B, 1]
    s_next     = estimator.sample(obs, action)               # [B, obs_dim]
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import utils.utils as utils
# --------------------------------------------------------------------------- #
#  Neural network                                                               #
# --------------------------------------------------------------------------- #
class TransitionModel(nn.Module):
    """Diagonal-Gaussian dynamics model.
    Maps (s, a) → (μ, σ) where both have shape [batch, obs_dim].
    Log-std is clamped to [log_std_min, log_std_max] for numerical stability.
    """
    LOG_STD_MIN = -4.0
    LOG_STD_MAX = +2.0
    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        # shared trunk → outputs (mean ‖ log_std) concatenated
        self.trunk = utils.mlp(
            input_dim=obs_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * obs_dim,
            hidden_depth=hidden_depth,
        )
        self.apply(utils.weight_init)
    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Return the predictive distribution N(μ, diag(σ²)).
        Args:
            obs:    [B, obs_dim]
            action: [B, action_dim]  (float; pass one-hot for discrete envs)
        Returns:
            dist: torch.distributions.Normal  (batch_shape=[B], event_shape=[obs_dim])
        """
        x = torch.cat([obs, action], dim=-1)
        mu, log_std = self.trunk(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mu, log_std.exp())
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor,
                 next_obs: torch.Tensor) -> torch.Tensor:
        """Sum log-prob over state dimensions → shape [B, 1]."""
        dist = self.forward(obs, action)
        return dist.log_prob(next_obs).sum(dim=-1, keepdim=True)
    def sample(self, obs: torch.Tensor,
               action: torch.Tensor) -> torch.Tensor:
        """Sample a next state from the predictive distribution."""
        dist = self.forward(obs, action)
        return dist.rsample()
    def mean(self, obs: torch.Tensor,
             action: torch.Tensor) -> torch.Tensor:
        """Deterministic (mean) prediction of next state."""
        dist = self.forward(obs, action)
        return dist.mean
# --------------------------------------------------------------------------- #
#  Training wrapper                                                             #
# --------------------------------------------------------------------------- #
class TransitionEstimator:
    """Wrapper that manages training of :class:`TransitionModel`.
    Parameters
    ----------
    obs_dim, action_dim:
        Dimensionalities of observation and action spaces.
    args:
        Hydra config object.  Only ``args.device`` is required; optional
        keys ``args.transition.lr``, ``args.transition.hidden_dim``, and
        ``args.transition.hidden_depth`` override defaults when present.
    device:
        torch.device (inferred from args when not supplied).
    """
    def __init__(self, obs_dim: int, action_dim: int, args, device=None):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.args       = args
        self.device     = torch.device(args.device) if device is None else device
        # Optional sub-config for transition model hyper-parameters
        t_cfg = getattr(args, "transition", None)
        lr           = float(getattr(t_cfg, "lr",           3e-4))
        hidden_dim   = int(  getattr(t_cfg, "hidden_dim",   256))
        hidden_depth = int(  getattr(t_cfg, "hidden_depth", 2))
        self.model = TransitionModel(
            obs_dim, action_dim, hidden_dim, hidden_depth
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.train()
    # ---------------------------------------------------------------------- #
    def train(self, training: bool = True):
        self.training = training
        self.model.train(training)
    # ---------------------------------------------------------------------- #
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor,
                 next_obs: torch.Tensor) -> torch.Tensor:
        """Log p(next_obs | obs, action), shape [B, 1]."""
        return self.model.log_prob(obs, action, next_obs)
    def sample(self, obs: torch.Tensor,
               action: torch.Tensor) -> torch.Tensor:
        """Sample s' ~ T(·|s, a), shape [B, obs_dim]."""
        with torch.no_grad():
            return self.model.sample(obs, action)
    def predict(self, obs: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """Deterministic mean prediction of s', shape [B, obs_dim]."""
        with torch.no_grad():
            return self.model.mean(obs, action)
    # ---------------------------------------------------------------------- #
    def update(self, obs: torch.Tensor, action: torch.Tensor,
               next_obs: torch.Tensor, logger=None, step: int = 0) -> dict:
        """One gradient step minimising the NLL loss on a transition batch.
        Args:
            obs, action, next_obs: tensors already on ``self.device``.
            logger: optional Logger instance (same interface as in sac.py).
            step:   global step counter for logging.
        Returns:
            Dictionary of scalar loss values.
        """
        # Negative log-likelihood: -E[log p(s'|s,a)]
        nll_loss = -self.model.log_prob(obs, action, next_obs).mean()
        self.optimizer.zero_grad()
        nll_loss.backward()
        self.optimizer.step()
        loss_dict = {"transition/nll_loss": nll_loss.item()}
        if logger is not None:
            logger.log("train/transition_nll", nll_loss, step)
        return loss_dict
    def update_from_buffer(self, replay_buffer, batch_size: int,
                           logger=None, step: int = 0) -> dict:
        """Convenience method: sample a batch from a Memory buffer and update.
        Args:
            replay_buffer: instance of ``dataset.memory.Memory``.
            batch_size:    number of transitions to sample.
        """
        obs, next_obs, action, _, _ = replay_buffer.get_samples(
            batch_size, self.device
        )
        return self.update(obs, action, next_obs, logger, step)
    # ---------------------------------------------------------------------- #
    def save(self, path: str, suffix: str = ""):
        """Save model weights to ``{path}{suffix}_transition``."""
        save_path = f"{path}{suffix}_transition"
        torch.save(self.model.state_dict(), save_path)
        print(f"Transition model saved → {save_path}")
    def load(self, path: str, suffix: str = ""):
        """Load model weights from ``{path}{suffix}_transition``."""
        load_path = f"{path}{suffix}_transition"
        self.model.load_state_dict(
            torch.load(load_path, map_location=self.device)
        )
        print(f"Transition model loaded ← {load_path}")

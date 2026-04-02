"""DiceAgent — Weighted Behavior Cloning policy extraction on top of a
trained MaxQ critic.

Two-phase workflow
==================
1. **Q-training**: use the standard ``MaxQ`` agent to learn Q(s,a) via
   IQ-Learn with the ``dice`` loss in an offline setting.
2. **Weighted BC** (this class): given the frozen Q-network, train a
   separate policy network π_θ(a|s) by solving

   .. math::

       \\max_\\theta \\; \\mathbb{E}_{s,a \\sim d^O}
       \\bigl[\\max\\{0,(f')^{-1}(r)\\}\\,\\log\\pi_\\theta(a|s)\\bigr]

   where ``r = (Q(s,a) - γV(s')) / α``  (dice-transformed Bellman
   residual).  After this phase ``choose_action`` automatically uses
   the BC actor.

Typical usage inside ``train_iq.py``::

    # Q-training with plain MaxQ …
    # … after Q converges:
    dice_agent = DiceAgent.from_maxq(agent)
    dice_agent.train_weighted_bc(expert_buffer, args, writer=writer)
    evaluate(dice_agent, eval_env)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

from wrappers.atari_wrapper import LazyFrames
from agent.maxq import MaxQ


# ------------------------------------------------------------------ #
#  Policy network
# ------------------------------------------------------------------ #

class PolicyNetwork(nn.Module):
    """Simple MLP policy for discrete actions."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def get_log_prob(self, obs, action):
        logits = self.forward(obs)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs.gather(1, action.long())


# ------------------------------------------------------------------ #
#  DiceAgent
# ------------------------------------------------------------------ #

class DiceAgent(MaxQ):
    """MaxQ + weighted-BC policy extraction for the dice loss setting."""

    def __init__(self, num_inputs, action_dim, batch_size, args):
        super().__init__(num_inputs, action_dim, batch_size, args)
        self.obs_dim = num_inputs

    # ----- factory: build from an already-trained MaxQ -------------- #

    @classmethod
    def from_maxq(cls, maxq_agent):
        """Create a DiceAgent that shares the trained Q-network weights
        of an existing ``MaxQ`` instance.

        The Q-net, target-net, transition model, and ensemble are
        **copied by value** (state-dict), so the original ``MaxQ`` is
        not affected.
        """
        args = maxq_agent.args
        obs_dim = args.agent.obs_dim
        action_dim = maxq_agent.action_dim
        batch_size = maxq_agent.batch_size

        agent = cls(obs_dim, action_dim, batch_size, args)

        agent.q_net.load_state_dict(maxq_agent.q_net.state_dict())
        agent.target_net.load_state_dict(maxq_agent.target_net.state_dict())
        agent.transition = maxq_agent.transition
        agent.ensemble = maxq_agent.ensemble
        return agent

    # ----- density-ratio computation -------------------------------- #

    @staticmethod
    def compute_density_ratio(reward, div):
        r"""Return :math:`\max\{0,\;(f')^{-1}(\text{reward})\}`.

        Parameters
        ----------
        reward : Tensor
            Dice-transformed Bellman residual ``(Q - γV') / α``.
        div : str or None
            Name of the f-divergence (matches ``args.method.div``).
        """
        if div == "hellinger":
            ratio = 1.0 / (1.0 - reward).clamp(min=1e-8) ** 2
        elif div in ("kl", "kl2"):
            ratio = torch.exp(reward)
        elif div == "kl_fix":
            ratio = 1.0 / (1.0 - reward).clamp(min=1e-8)
        elif div == "js":
            exp_r = torch.exp(reward)
            ratio = exp_r / (2.0 - exp_r).clamp(min=1e-8)
        elif div == "chi":
            ratio = reward + 1.0
        else:
            ratio = reward + 1.0
        return torch.clamp(ratio, min=0.0)

    # ----- dice reward (mirrors iq.py constrain → dice path) -------- #

    def _dice_reward(self, obs, next_obs, action, done, alpha):
        """Compute ``(Q(s,a) - γV(s')) / α`` following the sign
        convention in ``iq.py`` constrain → dice block."""
        current_Q = self.critic(obs, action)
        next_v = self.get_targetV(next_obs)
        y = (1 - done) * self.gamma * next_v

        reward = current_Q - y

        ensemble = getattr(self, "ensemble", None)
        if ensemble is not None:
            lambda_pen = getattr(self.args.method, "lambda_penalty", 0.0)
            if lambda_pen > 0:
                action_oh = self._action_onehot(action)
                pen = ensemble.penalty(obs, action_oh)
                reward = -reward - lambda_pen * pen
            else:
                reward = -reward
        else:
            reward = -reward

        reward = -reward / alpha
        return reward

    # ----- weighted BC training ------------------------------------- #

    def train_weighted_bc(self, expert_buffer, args,
                          logger=None, writer=None):
        r"""Train a policy network via weighted Behavior Cloning.

        Hyperparameters (read from ``args.method`` with defaults):

        ================ ======= ====================================
        key              default meaning
        ================ ======= ====================================
        bc_steps         10000   gradient steps for BC
        bc_lr            3e-4    actor learning rate
        bc_batch         256     mini-batch size
        bc_log_interval  500     print / tensorboard log frequency
        bc_hidden_dim    128     hidden-layer width of the actor MLP
        ================ ======= ====================================
        """
        bc_steps = int(getattr(args.method, "bc_steps", 10000))
        bc_lr = float(getattr(args.method, "bc_lr", 3e-4))
        bc_batch = int(getattr(args.method, "bc_batch", 256))
        bc_log_interval = int(getattr(args.method, "bc_log_interval", 500))
        hidden_dim = int(getattr(args.method, "bc_hidden_dim", 128))
        alpha = args.method.alpha
        div = args.method.div

        self.actor = PolicyNetwork(
            self.obs_dim, self.action_dim, hidden_dim
        ).to(self.device)
        actor_optimizer = Adam(self.actor.parameters(), lr=bc_lr)

        self.q_net.eval()
        self.target_net.eval()

        for step in range(1, bc_steps + 1):
            obs, next_obs, action, _, done = expert_buffer.get_samples(
                bc_batch, self.device)

            with torch.no_grad():
                reward = self._dice_reward(obs, next_obs, action, done, alpha)
                weights = self.compute_density_ratio(reward, div)

            log_prob = self.actor.get_log_prob(obs, action)
            bc_loss = -(weights * log_prob).mean()

            actor_optimizer.zero_grad()
            bc_loss.backward()
            actor_optimizer.step()

            if step % bc_log_interval == 0:
                mean_w = weights.mean().item()
                print(f'  [Weighted BC] step {step}/{bc_steps}  '
                      f'loss={bc_loss.item():.4f}  mean_weight={mean_w:.4f}')
                if writer is not None:
                    writer.add_scalar('bc/loss', bc_loss.item(), step)
                    writer.add_scalar('bc/mean_weight', mean_w, step)

        self.q_net.train()
        self.target_net.train()
        print(f'Weighted BC training finished ({bc_steps} steps).')

    # ----- action selection ----------------------------------------- #

    def choose_action(self, state, sample=False):
        """Use BC actor if trained, otherwise fall back to MaxQ argmax."""
        if isinstance(state, LazyFrames):
            state = np.array(state) / 255.0

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if self.actor is not None:
                logits = self.actor(state)
                if sample:
                    action = Categorical(logits=logits).sample()
                else:
                    action = logits.argmax(dim=1)
            else:
                q_values = self.q_net(state)
                action = q_values.argmax(dim=1)

        return action.detach().cpu().numpy()[0]

    # ----- save / load ---------------------------------------------- #

    def save(self, path, suffix=""):
        super().save(path, suffix)
        if self.actor is not None:
            torch.save(self.actor.state_dict(), f"{path}{suffix}_actor")

    def load(self, path, suffix=""):
        super().load(path, suffix)
        actor_path = f'{path}/{self.args.agent.name}{suffix}_actor'
        if os.path.isfile(actor_path):
            hidden_dim = int(getattr(
                getattr(self.args, "method", None), "bc_hidden_dim", 128))
            self.actor = PolicyNetwork(
                self.obs_dim, self.action_dim, hidden_dim
            ).to(self.device)
            self.actor.load_state_dict(
                torch.load(actor_path, map_location=self.device))
            print(f'Loaded BC actor from {actor_path}')

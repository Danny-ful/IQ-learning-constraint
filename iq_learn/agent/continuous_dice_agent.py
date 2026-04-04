"""ContinuousDiceAgent — Weighted Behavior Cloning policy extraction on top of
a trained SAC critic for **continuous** action spaces.

Two-phase workflow
==================
1. **Q-training**: use the standard ``SAC`` agent to learn Q(s,a) via
   IQ-Learn with the ``dice`` loss.  The SAC actor is still updated so
   that ``getV`` / ``get_targetV`` produce reasonable soft-value
   estimates throughout training.
2. **Weighted BC** (this class): given the frozen twin-Q network, train
   a separate Gaussian policy π_θ(a|s) by solving

   .. math::

       \\max_\\theta \\; \\mathbb{E}_{s,a \\sim d^O}
       \\bigl[\\max\\{0,(f')^{-1}(r)\\}\\,\\log\\pi_\\theta(a|s)\\bigr]

   where ``r = (Q(s,a) - γV(s')) / α``  (dice-transformed Bellman
   residual).  After this phase ``choose_action`` automatically routes
   to the BC actor.

Typical usage inside ``train_iq.py``::

    # Q-training with plain SAC …
    # … after Q converges:
    dice_agent = ContinuousDiceAgent.from_sac(agent)
    dice_agent.train_weighted_bc(expert_buffer, args, writer=writer)
    evaluate(dice_agent, eval_env)
"""

import os
import numpy as np
import torch
from torch.optim import Adam
from tqdm.auto import tqdm

from agent.sac import SAC
from agent.sac_models import DiagGaussianActor
from agent.dice_agent import DiceAgent


class ContinuousDiceAgent(SAC):
    """SAC + weighted-BC policy extraction for the dice loss setting
    in continuous action spaces."""

    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        super().__init__(obs_dim, action_dim, action_range, batch_size, args)
        self.obs_dim = obs_dim
        self.bc_actor = None

    # ----- factory: build from an already-trained SAC ----------------- #

    @classmethod
    def from_sac(cls, sac_agent):
        """Create a ContinuousDiceAgent that copies the trained networks
        of an existing ``SAC`` instance.

        Critic, critic-target, SAC actor, and temperature are
        **copied by value** (state-dict), so the original ``SAC`` is
        not affected.
        """
        args = sac_agent.args
        obs_dim = args.agent.obs_dim
        action_dim = args.agent.action_dim
        action_range = sac_agent.action_range
        batch_size = sac_agent.batch_size

        agent = cls(obs_dim, action_dim, action_range, batch_size, args)

        agent.critic.load_state_dict(sac_agent.critic.state_dict())
        agent.critic_target.load_state_dict(
            sac_agent.critic_target.state_dict())
        agent.actor.load_state_dict(sac_agent.actor.state_dict())
        agent.log_alpha = sac_agent.log_alpha.clone().detach()
        agent.log_alpha.requires_grad_(True)
        return agent

    # ----- density-ratio computation (shared with discrete dice) ------ #

    compute_density_ratio = staticmethod(DiceAgent.compute_density_ratio)

    # ----- dice reward ------------------------------------------------ #

    def _dice_reward(self, obs, next_obs, action, done, alpha):
        """Compute ``(Q(s,a) - γV(s')) / α`` for continuous actions.

        Uses the SAC actor to evaluate V(s') via ``get_targetV``.
        """
        current_Q = self.critic(obs, action)
        next_v = self.get_targetV(next_obs)
        y = (1 - done) * self.gamma * next_v
        reward = (current_Q - y) / alpha
        return reward

    # ----- weighted BC training --------------------------------------- #

    def train_weighted_bc(self, buffer, args,
                          logger=None, writer=None):
        r"""Train a continuous Gaussian policy via weighted Behavior Cloning.

        Parameters
        ----------
        buffer : Memory
            Replay buffer to sample transitions from.  In online mode
            this is typically the expert buffer; in offline mode this
            should be the supplementary / offline dataset buffer.

        Hyperparameters (read from ``args.method`` with defaults):

        ================= ======= ====================================
        key               default meaning
        ================= ======= ====================================
        bc_steps          10000   gradient steps for BC
        bc_lr             3e-4    actor learning rate
        bc_batch          256     mini-batch size
        bc_log_interval   500     print / tensorboard log frequency
        bc_hidden_dim     256     hidden-layer width of the actor MLP
        bc_hidden_depth   2       hidden-layer depth of the actor MLP
        ================= ======= ====================================
        """
        bc_steps = int(getattr(args.method, "bc_steps", 10000))
        bc_lr = float(getattr(args.method, "bc_lr", 3e-4))
        bc_batch = int(getattr(args.method, "bc_batch", 256))
        bc_log_interval = int(getattr(args.method, "bc_log_interval", 500))
        hidden_dim = int(getattr(args.method, "bc_hidden_dim", 256))
        hidden_depth = int(getattr(args.method, "bc_hidden_depth", 2))
        alpha = args.method.alpha
        div = args.method.div

        action_dim = args.agent.action_dim

        self.bc_actor = DiagGaussianActor(
            self.obs_dim, action_dim, hidden_dim, hidden_depth,
            log_std_bounds=[-5, 2],
        ).to(self.device)
        bc_optimizer = Adam(self.bc_actor.parameters(), lr=bc_lr)

        self.critic.eval()
        self.critic_target.eval()
        self.actor.eval()

        bc_pbar = tqdm(
            range(1, bc_steps + 1),
            desc="Weighted BC",
            dynamic_ncols=True,
        )

        for step in bc_pbar:
            obs, next_obs, action, _, done = buffer.get_samples(
                bc_batch, self.device)

            with torch.no_grad():
                reward = self._dice_reward(obs, next_obs, action, done, alpha)
                ### ensure reward is in valid domain of (f')^{-1}
                reward = DiceAgent.project_reward_to_valid_domain(reward, div, eps=1e-6)
                weights = self.compute_density_ratio(reward, div)

            dist = self.bc_actor(obs)
            action_clamped = action.clamp(-0.999, 0.999)
            log_prob = dist.log_prob(action_clamped).sum(-1, keepdim=True)

            bc_loss = -(weights * log_prob).mean()

            bc_optimizer.zero_grad()
            bc_loss.backward()
            bc_optimizer.step()

            if step % bc_log_interval == 0:
                mean_w = weights.mean().item()
                bc_pbar.set_postfix(
                    loss=f"{bc_loss.item():.4f}",
                    mean_weight=f"{mean_w:.4f}",
                )
                print(f'  [Weighted BC] step {step}/{bc_steps}  '
                      f'loss={bc_loss.item():.4f}  mean_weight={mean_w:.4f}')
                if writer is not None:
                    writer.add_scalar('bc/loss', bc_loss.item(), step)
                    writer.add_scalar('bc/mean_weight', mean_w, step)

        bc_pbar.close()
        self.critic.train()
        self.critic_target.train()
        self.actor.train()
        print(f'Weighted BC training finished ({bc_steps} steps).')

    # ----- action selection ------------------------------------------- #

    def choose_action(self, state, sample=False):
        """Use BC actor if trained, otherwise fall back to SAC actor."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if self.bc_actor is not None:
                dist = self.bc_actor(state)
            else:
                dist = self.actor(state)
            action = dist.sample() if sample else dist.mean

        return action.detach().cpu().numpy()[0]

    # ----- save / load ------------------------------------------------ #

    def save(self, path, suffix=""):
        super().save(path, suffix)
        if self.bc_actor is not None:
            torch.save(self.bc_actor.state_dict(),
                       f"{path}{suffix}_bc_actor")

    def load(self, path, suffix=""):
        super().load(path, suffix)
        bc_path = f'{path}/{self.args.agent.name}{suffix}_bc_actor'
        if os.path.isfile(bc_path):
            action_dim = self.args.agent.action_dim
            hidden_dim = int(getattr(
                getattr(self.args, "method", None), "bc_hidden_dim", 256))
            hidden_depth = int(getattr(
                getattr(self.args, "method", None), "bc_hidden_depth", 2))
            self.bc_actor = DiagGaussianActor(
                self.obs_dim, action_dim, hidden_dim, hidden_depth,
                log_std_bounds=[-5, 2],
            ).to(self.device)
            self.bc_actor.load_state_dict(
                torch.load(bc_path, map_location=self.device))
            print(f'Loaded BC actor from {bc_path}')

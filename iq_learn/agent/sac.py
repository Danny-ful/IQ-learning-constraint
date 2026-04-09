import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra

from utils.utils import soft_update
from agent.penalty import TransitionEnsemble


def _is_finite_tensor(tensor):
    return bool(torch.isfinite(tensor).all().item())


def _safe_backward_step(optimizer, params, loss, max_grad_norm, logger=None, step=None, prefix=None):
    if not _is_finite_tensor(loss):
        if logger is not None and step is not None and prefix is not None:
            logger.log(f'{prefix}/non_finite_loss', 1.0, step)
        optimizer.zero_grad(set_to_none=True)
        return False, None

    optimizer.zero_grad()
    loss.backward()

    params = [param for param in params if param.requires_grad]
    grads = [param.grad for param in params if param.grad is not None]
    if any(not torch.isfinite(grad).all() for grad in grads):
        if logger is not None and step is not None and prefix is not None:
            logger.log(f'{prefix}/non_finite_grad', 1.0, step)
        optimizer.zero_grad(set_to_none=True)
        return False, None

    grad_norm = None
    if grads and max_grad_norm is not None and max_grad_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if not _is_finite_tensor(torch.as_tensor(grad_norm)):
            if logger is not None and step is not None and prefix is not None:
                logger.log(f'{prefix}/non_finite_grad_norm', 1.0, step)
            optimizer.zero_grad(set_to_none=True)
            return False, None

    optimizer.step()

    if grad_norm is not None and logger is not None and step is not None and prefix is not None:
        logger.log(f'{prefix}/grad_norm', grad_norm, step)

    return True, grad_norm


class SAC(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent

        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.learn_temp
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        # _recursive_=False: do not recurse into `args` (it contains critic_cfg -> q_net again).
        self.critic = hydra.utils.instantiate(
            agent_cfg.critic_cfg, args=args, _recursive_=False).to(self.device)

        self.critic_target = hydra.utils.instantiate(
            agent_cfg.critic_cfg, args=args, _recursive_=False).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=agent_cfg.actor_lr,
                                    betas=agent_cfg.actor_betas)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas)
        self.log_alpha_optimizer = Adam([self.log_alpha],
                                        lr=agent_cfg.alpha_lr,
                                        betas=agent_cfg.alpha_betas)

        ensemble_k = int(getattr(getattr(args, "method", None),
                                 "ensemble_k", 0))
        if ensemble_k > 0 and bool(getattr(args.method, "penalty", False)):
            self.ensemble = TransitionEnsemble(
                ensemble_k, obs_dim, action_dim, args, self.device,
                continuous=True)
        else:
            self.ensemble = None

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.critic_target

    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        # assert action.ndim == 2 and action.shape[0] == 1
        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        current_Q = self.critic(obs, action)
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V

    def get_targetV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        target_Q = self.critic_target(obs, action)
        target_V = target_Q - self.alpha.detach() * log_prob
        return target_V

    def update(self, replay_buffer, logger, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device)

        losses = self.update_critic(obs, action, reward, next_obs, done,
                                    logger, step)

        if step % self.actor_update_frequency == 0:
            actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
            losses.update(actor_alpha_losses)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic, self.critic_target,
                        self.critic_tau)

        return losses

    def update_critic(self, obs, action, reward, next_obs, done, logger, step):

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)

            target_Q = self.critic_target(next_obs, next_action)
            target_V = target_Q - self.alpha.detach() * log_prob
            target_Q = reward + (1 - done) * self.gamma * target_V

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss = F.mse_loss(current_Q1, target_Q)
        q2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = q1_loss + q2_loss

        # Optimize the critic
        critic_grad_clip = float(getattr(self.args.agent, "critic_grad_clip", 10.0))
        stepped, _ = _safe_backward_step(
            self.critic_optimizer,
            self.critic.parameters(),
            critic_loss,
            critic_grad_clip,
            logger=logger,
            step=step,
            prefix='train/critic',
        )
        if not stepped:
            logger.log('train/critic_skipped_step', 1.0, step)

        # self.critic.log(logger, step)
        return {
            'critic_loss/critic_1': q1_loss.item(),
            'critic_loss/critic_2': q2_loss.item(),
            'loss/critic': critic_loss.item()}

    def update_actor_and_alpha(self, obs, logger, step):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train/actor_loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/actor_entropy', -log_prob.mean(), step)

        # optimize the actor
        actor_grad_clip = float(getattr(self.args.agent, "actor_grad_clip", 5.0))
        stepped, _ = _safe_backward_step(
            self.actor_optimizer,
            self.actor.parameters(),
            actor_loss,
            actor_grad_clip,
            logger=logger,
            step=step,
            prefix='train/actor',
        )
        if not stepped:
            logger.log('train/actor_skipped_step', 1.0, step)

        losses = {
            'loss/actor': actor_loss.item(),
            'actor_loss/target_entropy': self.target_entropy,
            'actor_loss/entropy': -log_prob.mean().item()}

        # self.actor.log(logger, step)
        if self.learn_temp:
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train/alpha_loss', alpha_loss, step)
            logger.log('train/alpha_value', self.alpha, step)
            stepped, _ = _safe_backward_step(
                self.log_alpha_optimizer,
                [self.log_alpha],
                alpha_loss,
                actor_grad_clip,
                logger=logger,
                step=step,
                prefix='train/alpha',
            )
            if not stepped:
                logger.log('train/alpha_skipped_step', 1.0, step)

            losses.update({
                'alpha_loss/loss': alpha_loss.item(),
                'alpha_loss/value': self.alpha.item(),
            })
        return losses

    # Save model parameters
    def save(self, path, suffix=""):
        actor_path = f"{path}{suffix}_actor"
        critic_path = f"{path}{suffix}_critic"

        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        actor_path = f'{path}/{self.args.agent.name}{suffix}_actor'
        critic_path = f'{path}/{self.args.agent.name}{suffix}_critic'
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()

    def sample_actions(self, obs, num_actions):
        """For CQL style training."""
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0] * num_actions, obs.shape[1])
        action, log_prob, _ = self.actor.sample(obs_temp)
        return action, log_prob.view(obs.shape[0], num_actions, 1)

    def _get_tensor_values(self, obs, actions, network=None):
        """For CQL style training."""
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def cqlV(self, obs, network, num_random=10):
        """For CQL style training."""
        # importance sampled version
        action, log_prob = self.sample_actions(obs, num_random)
        current_Q = self._get_tensor_values(obs, action, network)

        random_action = torch.FloatTensor(
            obs.shape[0] * num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)

        random_density = np.log(0.5 ** action.shape[-1])
        rand_Q = self._get_tensor_values(obs, random_action, network)
        alpha = self.alpha.detach()

        cat_Q = torch.cat(
            [rand_Q - alpha * random_density, current_Q - alpha * log_prob.detach()], 1
        )
        cql_V = torch.logsumexp(cat_Q / alpha, dim=1).mean() * alpha
        return cql_V

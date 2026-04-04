import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra

from wrappers.atari_wrapper import LazyFrames
from agent.transition import TransitionEstimator
from agent.penalty import TransitionEnsemble


class MaxQ(object):
    def __init__(self, num_inputs, action_dim, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent
        
        self.actor = None
        self.action_dim = action_dim
        self.critic_tau = agent_cfg.critic_tau
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        # 初始化 Q 网络
        self.q_net = hydra.utils.instantiate(
            agent_cfg.critic_cfg, args=args, device=self.device, _recursive_=False).to(self.device)
        self.target_net = hydra.utils.instantiate(
            agent_cfg.critic_cfg, args=args, device=self.device, _recursive_=False).to(self.device)
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.critic_optimizer = Adam(self.q_net.parameters(), lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas)

        self.transition = TransitionEstimator(
            num_inputs, action_dim, args, self.device)

        ensemble_k = int(getattr(getattr(args, "method", None),
                                 "ensemble_k", 0))
        if ensemble_k > 0:
            self.ensemble = TransitionEnsemble(
                ensemble_k, num_inputs, action_dim, args, self.device)
        else:
            self.ensemble = None

        self.train()
        self.target_net.train()

    def train(self, training=True):
        self.training = training
        self.q_net.train(training)

    @property
    def critic_net(self):
        return self.q_net

    @property
    def critic_target_net(self):
        return self.target_net

    def choose_action(self, state, sample=False):
        """
        
        """
        if isinstance(state, LazyFrames):
            state = np.array(state) / 255.0
        
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_net(state)
            # 纯 Max-Q 逻辑
            action = q_values.argmax(dim=1)
            
        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        """V(s) = max_a Q(s, a)"""
        q = self.q_net(obs)
        v, _ = q.max(dim=1, keepdim=True)
        return v

    def get_targetV(self, obs):
        
        q = self.target_net(obs)
        target_v, _ = q.max(dim=1, keepdim=True)
        return target_v

    def critic(self, obs, action, both=False):
        
        q = self.q_net(obs, both)
        if isinstance(q, tuple) and both:
            q1, q2 = q
            critic1 = q1.gather(1, action.long())
            critic2 = q2.gather(1, action.long())
            return critic1, critic2

        return q.gather(1, action.long())

    def _action_onehot(self, action):
        """Convert integer actions [B,1] to one-hot float [B, action_dim]."""
        return F.one_hot(
            action.long().squeeze(1), self.action_dim
        ).float()

    def update(self, replay_buffer, logger, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device)

        action_oh = self._action_onehot(action)
        trans_losses = self.transition.update(
            obs, action_oh, next_obs, logger, step)

        if self.ensemble is not None:
            ens_losses = self.ensemble.update(
                obs, action_oh, next_obs, logger, step)
            trans_losses.update(ens_losses)

        losses = self.update_critic(obs, action, action_oh, reward, next_obs,
                                    done, logger, step)
        losses.update(trans_losses)

        if step % self.critic_target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return losses

    def update_critic(self, obs, action, action_oh, reward, next_obs,
                       done, logger, step):
        with torch.no_grad():
            predicted_next_obs = self.transition.predict(obs, action_oh)
            next_v = self.get_targetV(predicted_next_obs)
            y = reward + (1 - done) * self.gamma * next_v

        current_q = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q, y)

        if logger is not None:
            logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {'loss/critic': critic_loss.item()}

    def save(self, path, suffix=""):
        torch.save(self.q_net.state_dict(), f"{path}{suffix}")
        self.transition.save(path, suffix)
        if self.ensemble is not None:
            self.ensemble.save(path, suffix)

    def load(self, path, suffix=""):
        critic_path = f'{path}/{self.args.agent.name}{suffix}'
        self.q_net.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.transition.load(path, suffix)
        if self.ensemble is not None:
            self.ensemble.load(path, suffix)

    def infer_q(self, state, action):
        # 保持与原接口一致
        if isinstance(state, LazyFrames):
            state = np.array(state) / 255.0
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        if isinstance(state, LazyFrames):
            state = np.array(state) / 255.0
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()

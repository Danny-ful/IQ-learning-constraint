import torch
import torch.nn.functional as F

from agent.sac import SAC


class CQL(SAC):
    """Continuous-action CQL agent built on top of SAC networks."""

    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        super().__init__(obs_dim, action_dim, action_range, batch_size, args)
        method_cfg = args.method
        self.cql_n_actions = int(getattr(method_cfg, "cql_n_actions", 10))
        self.cql_alpha = float(getattr(method_cfg, "cql_alpha", 1.0))

    def _q1(self, obs, action):
        q1, _ = self.critic(obs, action, both=True)
        return q1

    def _q2(self, obs, action):
        _, q2 = self.critic(obs, action, both=True)
        return q2

    def _conservative_losses(self, obs, current_q1, current_q2):
        cql_v1 = self.cqlV(obs, self._q1, num_random=self.cql_n_actions)
        cql_v2 = self.cqlV(obs, self._q2, num_random=self.cql_n_actions)
        cql_loss1 = cql_v1 - current_q1.mean()
        cql_loss2 = cql_v2 - current_q2.mean()
        return cql_loss1, cql_loss2

    def update_critic(self, obs, action, reward, next_obs, done, logger, step):
        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)
            target_q = self.critic_target(next_obs, next_action)
            target_v = target_q - self.alpha.detach() * log_prob
            target_q = reward + (1 - done) * self.gamma * target_v

        current_q1, current_q2 = self.critic(obs, action, both=True)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        bellman_loss = q1_loss + q2_loss

        cql_loss1, cql_loss2 = self._conservative_losses(
            obs, current_q1, current_q2)
        cql_penalty = self.cql_alpha * (cql_loss1 + cql_loss2)
        critic_loss = bellman_loss + cql_penalty

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if logger is not None:
            logger.log('train/cql_loss_1', cql_loss1, step)
            logger.log('train/cql_loss_2', cql_loss2, step)
            logger.log('train/cql_penalty', cql_penalty, step)

        return {
            'critic_loss/critic_1': q1_loss.item(),
            'critic_loss/critic_2': q2_loss.item(),
            'loss/critic': critic_loss.item(),
            'loss/cql_1': cql_loss1.item(),
            'loss/cql_2': cql_loss2.item(),
            'loss/cql_penalty': cql_penalty.item(),
        }

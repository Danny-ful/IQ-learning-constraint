"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import (
    eval_mode,
    average_dicts,
    get_concat_samples,
    evaluate,
    soft_update,
    hard_update,
    gym_reset,
    gym_step,
    gym_maybe_seed,
)
from utils.logger import Logger
from iq import iq_loss
from agent.sac import SAC
from agent.dice_agent import DiceAgent
from agent.continuous_dice_agent import ContinuousDiceAgent

torch.set_num_threads(2)


def init_wandb_with_fallback(args, wandb_cfg):
    """Initialize wandb with graceful fallback for permission/network failures."""
    requested_mode = os.getenv("WANDB_MODE", "online")
    wandb_project = os.getenv("WANDB_PROJECT", args.project_name)
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_run_name = os.getenv("WANDB_NAME") or args.exp_name or None

    base_kwargs = dict(
        project=wandb_project,
        sync_tensorboard=True,
        reinit=True,
        config=wandb_cfg,
    )
    if wandb_entity:
        base_kwargs["entity"] = wandb_entity
    if wandb_run_name:
        base_kwargs["name"] = wandb_run_name

    if requested_mode in ("offline", "disabled"):
        wandb.init(mode=requested_mode, **base_kwargs)
        print(f"[wandb] Initialized in mode='{requested_mode}'.")
        return

    try:
        wandb.init(mode="online", **base_kwargs)
        print("[wandb] Online init succeeded.")
        return
    except Exception as err:
        print(f"[wandb] Online init failed: {err}")

    for mode in ("offline", "disabled"):
        try:
            wandb.init(mode=mode, **base_kwargs)
            print(f"[wandb] Fallback to mode='{mode}' succeeded.")
            return
        except Exception as err:
            print(f"[wandb] Fallback mode='{mode}' failed: {err}")

    print("[wandb] Disabled: all init attempts failed. Training will continue without wandb logging.")


def safe_wandb_set_best_returns(best_eval_returns):
    """Best-effort update for wandb summary."""
    run = getattr(wandb, "run", None)
    if run is None:
        return
    try:
        run.summary["best_returns"] = best_eval_returns
    except Exception as err:
        print(f"[wandb] Failed to update summary: {err}")


def safe_wandb_finish():
    """Best-effort wandb finish."""
    run = getattr(wandb, "run", None)
    if run is None:
        return
    try:
        wandb.finish()
    except Exception as err:
        print(f"[wandb] Failed to finish run: {err}")


def _make_dice_agent(agent):
    """Create the appropriate DiceAgent variant based on the base agent type.

    - Continuous (SAC-based)  → ContinuousDiceAgent
    - Discrete  (MaxQ-based)  → DiceAgent
    """
    if isinstance(agent, SAC):
        return ContinuousDiceAgent.from_sac(agent)
    return DiceAgent.from_maxq(agent)


def sync_progress_bar(progress_bar, step, **postfix):
    """Advance a tqdm bar to an absolute step count."""
    if progress_bar is None:
        return

    delta = max(0, step - progress_bar.n)
    if delta:
        progress_bar.update(delta)

    if postfix:
        progress_bar.set_postfix(postfix)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    if args.method.loss == "dice" and not args.offline:
        raise ValueError("method.loss=dice is only supported when offline=True")

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    env = make_env(args)
    eval_env = make_env(args)

    # Seed envs (gym>=0.26 removed env.seed; use reset(seed=) via gym_maybe_seed)
    first_obs = gym_maybe_seed(env, args.seed)
    gym_maybe_seed(eval_env, args.seed + 10)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)
    INITIAL_STATES = 128  # Num initial states to use to calculate value of initial state distribution s_0

    agent = make_agent(env, args)

    # After make_agent, obs_dim/action_dim are set so interpolations resolve; plain dict for wandb JSON.
    wandb_cfg = OmegaConf.to_container(args, resolve=True)
    init_wandb_with_fallback(args, wandb_cfg)

    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent.load(pretrain_path)
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    # Load expert data
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

    # Load offline / supplementary data when running in offline mode
    if args.offline:
        # Always mix expert data into offline buffer.
        online_memory_replay.load(
            hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
            num_trajs=args.expert.demos,
            sample_freq=args.expert.subsample_freq,
            seed=args.seed + 43)

        # Also mix all datasets under iq_learn/supplement into offline buffer.
        supplement_dir = hydra.utils.to_absolute_path("supplement")
        if os.path.isdir(supplement_dir):
            supplement_files = sorted(
                f for f in os.listdir(supplement_dir)
                if os.path.isfile(os.path.join(supplement_dir, f))
                and f.endswith((".pkl", ".npy", ".pt")))
            for idx, supplement_file in enumerate(supplement_files):
                online_memory_replay.load(
                    os.path.join(supplement_dir, supplement_file),
                    num_trajs=getattr(args.expert, 'offline_demos', -1),
                    sample_freq=args.expert.subsample_freq,
                    seed=args.seed + 44 + idx)
                print(f'--> Loaded Supplement dataset: {supplement_file}')

        print(f'--> Offline buffer size (expert + Supplement): {online_memory_replay.size()}')

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir,
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=args.agent.name)

    best_eval_returns = -np.inf

    # ------------------------------------------------------------------ #
    #  Offline training: no environment interaction                       #
    # ------------------------------------------------------------------ #
    if args.offline:
        agent.iq_update = types.MethodType(iq_update, agent)
        agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
        offline_eval_episode = 0
        offline_pbar = tqdm(
            range(1, LEARN_STEPS + 1),
            desc="Offline IQ training",
            dynamic_ncols=True,
        )

        for learn_step in offline_pbar:
            losses = agent.iq_update(
                online_memory_replay, expert_memory_replay,
                logger, learn_step)

            if learn_step % args.log_interval == 0:
                for key, loss in losses.items():
                    writer.add_scalar(key, loss, global_step=learn_step)
                critic_loss = losses.get('critic_loss')
                if critic_loss is not None:
                    offline_pbar.set_postfix(
                        critic=f"{float(critic_loss):.4f}",
                        best=f"{best_eval_returns:.2f}" if np.isfinite(best_eval_returns) else "N/A",
                    )

            if learn_step % int(args.env.eval_interval) == 0:
                if args.method.loss == "dice":
                    eval_dice = _make_dice_agent(agent)
                    eval_dice.train_weighted_bc(
                        buffer=online_memory_replay, args=args,
                        logger=logger, writer=writer)
                    eval_returns, _ = evaluate(
                        eval_dice, eval_env, num_episodes=args.eval.eps)
                else:
                    eval_returns, _ = evaluate(
                        agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                offline_eval_episode += 1
                logger.log('eval/episode_reward', returns, learn_step)
                logger.log('eval/episode', offline_eval_episode, learn_step)
                logger.dump(learn_step, ty='eval')
                if returns > best_eval_returns:
                    best_eval_returns = returns
                    safe_wandb_set_best_returns(best_eval_returns)
                    if args.method.loss == "dice":
                        save(eval_dice, 0, args, output_dir='results_best')
                    else:
                        save(agent, 0, args, output_dir='results_best')
                offline_pbar.set_postfix(
                    eval=f"{returns:.2f}",
                    best=f"{best_eval_returns:.2f}",
                )

        offline_pbar.close()
        print('Offline Q-training finished!')

        if args.method.loss == "dice":
            print('Starting Weighted BC policy extraction...')
            dice_agent = _make_dice_agent(agent)
            dice_agent.train_weighted_bc(
                buffer=online_memory_replay, args=args,
                logger=logger, writer=writer)
            eval_returns, _ = evaluate(
                dice_agent, eval_env, num_episodes=args.eval.eps)
            bc_returns = np.mean(eval_returns)
            print(f'Weighted BC eval returns: {bc_returns:.2f}')
            offline_eval_episode += 1
            logger.log('eval/episode_reward', bc_returns, LEARN_STEPS)
            logger.log('eval/bc_episode_reward', bc_returns, LEARN_STEPS)
            logger.log('eval/episode', offline_eval_episode, LEARN_STEPS)
            logger.dump(LEARN_STEPS, ty='eval')
            save(dice_agent, 0, args, output_dir='results_bc')

        save(agent, 0, args, output_dir='results')
        safe_wandb_finish()
        return

    # ------------------------------------------------------------------ #
    #  Online training: interact with environment                         #
    # ------------------------------------------------------------------ #
    steps = 0
    scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards

    learn_steps = 0
    begin_learn = False
    episode_reward = 0
    online_pbar = tqdm(
        total=LEARN_STEPS,
        desc="Online IQ training",
        dynamic_ncols=True,
    )

    # Sample initial states from env
    if first_obs is not None:
        state_0 = [first_obs] * INITIAL_STATES
    else:
        state_0 = [gym_reset(env)] * INITIAL_STATES
    if isinstance(state_0[0], LazyFrames):
        state_0 = np.array(state_0) / 255.0
    state_0 = torch.FloatTensor(np.array(state_0)).to(args.device)

    for epoch in count():
        state = gym_reset(env)
        episode_reward = 0
        done = False

        start_time = time.time()
        for episode_step in range(EPISODE_STEPS):

            if steps < args.num_seed_steps:
                # Seed replay buffer with random actions
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = gym_step(env, action)
            episode_reward += reward
            steps += 1

            if learn_steps % args.env.eval_interval == 0:
                eval_returns, eval_timesteps = evaluate(
                    agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                learn_steps += 1  # To prevent repeated eval at timestep 0
                sync_progress_bar(
                    online_pbar,
                    learn_steps,
                    eval=f"{returns:.2f}",
                    best=f"{best_eval_returns:.2f}" if np.isfinite(best_eval_returns) else "N/A",
                )
                logger.log('eval/episode_reward', returns, learn_steps)
                logger.log('eval/episode', epoch, learn_steps)
                logger.dump(learn_steps, ty='eval')

                if returns > best_eval_returns:
                    best_eval_returns = returns
                    safe_wandb_set_best_returns(best_eval_returns)
                    save(agent, epoch, args, output_dir='results_best')

            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            online_memory_replay.add((state, next_state, action, reward, done_no_lim))

            if online_memory_replay.size() > INITIAL_MEMORY:
                # Start learning
                if begin_learn is False:
                    print('Learn begins!')
                    begin_learn = True

                learn_steps += 1
                sync_progress_bar(
                    online_pbar,
                    learn_steps,
                    reward=f"{episode_reward:.2f}",
                    best=f"{best_eval_returns:.2f}" if np.isfinite(best_eval_returns) else "N/A",
                )
                if learn_steps == LEARN_STEPS:
                    print('Q-training finished!')
                    save(agent, epoch, args, output_dir='results')
                    online_pbar.close()
                    safe_wandb_finish()
                    return

                ######
                # IQ-Learn Modification
                agent.iq_update = types.MethodType(iq_update, agent)
                agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
                losses = agent.iq_update(online_memory_replay,
                                         expert_memory_replay, logger, learn_steps)
                ######

                if learn_steps % args.log_interval == 0:
                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)

            if done:
                break
            state = next_state

        rewards_window.append(episode_reward)
        logger.log('train/episode', epoch, learn_steps)
        logger.log('train/episode_reward', episode_reward, learn_steps)
        logger.log('train/duration', time.time() - start_time, learn_steps)
        logger.dump(learn_steps, save=begin_learn)
        sync_progress_bar(
            online_pbar,
            learn_steps,
            reward=f"{episode_reward:.2f}",
            best=f"{best_eval_returns:.2f}" if np.isfinite(best_eval_returns) else "N/A",
        )
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(agent, epoch, args, output_dir='results')


def save(agent, epoch, args, output_dir='results'):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'sqil_{args.env.name}'
        else:
            name = f'iq_{args.env.name}'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{args.agent.name}_{name}')


# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    loss_dict = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    current_Q = self.critic(obs, action)
    y = (1 - done) * self.gamma * self.getV(next_obs)
    if args.train.use_target:
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss


def iq_update_critic(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
        q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()
    return loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch, logger, step)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            if self.args.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


if __name__ == "__main__":
    main()

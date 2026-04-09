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
from agent.cql import CQL
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


def _make_dice_agent(agent, args):
    """Create the appropriate DiceAgent variant based on the base agent type.

    - Continuous (SAC-based)  → ContinuousDiceAgent
    - Discrete  (MaxQ-based)  → DiceAgent
    """
    if isinstance(agent, SAC):
        if getattr(args.method, "dice_use_cql_q", False) and not isinstance(agent, CQL):
            raise ValueError(
                "method.dice_use_cql_q=True should have forced a CQL base agent."
            )
        return ContinuousDiceAgent.from_sac(agent)
    return DiceAgent.from_maxq(agent)


def _warm_start_eval_bc(current_dice_agent, previous_dice_agent, args):
    """Warm-start eval-time weighted BC from the previous eval policy."""
    if previous_dice_agent is None:
        return False
    if not bool(getattr(args.method, "bc_warm_start_eval", True)):
        return False
    return current_dice_agent.warm_start_bc_from(previous_dice_agent)


def _log_weight_stats(logger, dice_agent, buffer, args, step, prefix="eval/bc_weight"):
    """Log sampled weighted-BC weight statistics for degeneration checks."""
    stats, histograms = dice_agent.estimate_weight_stats(buffer, args)
    if not stats:
        return

    summary_keys = (
        "ess_ratio",
        "clip_frac",
        "raw/p99",
        "normalized/std",
    )
    for key in summary_keys:
        value = stats.get(key)
        if value is None:
            continue
        logger.log(f"{prefix}/{key}", value, step)

    print(
        "[dice_mode] BC weight summary "
        f"(step={step}): "
        f"ess_ratio={stats.get('ess_ratio', float('nan')):.4f}  "
        f"norm_std={stats.get('normalized/std', float('nan')):.4f}  "
        f"raw_p99={stats.get('raw/p99', float('nan')):.4f}  "
        f"clip_frac={stats.get('clip_frac', float('nan')):.4f}"
    )

    if bool(getattr(args.method, "bc_weight_eval_hist", True)):
        for key, values in histograms.items():
            logger.log_histogram(f"{prefix}/{key}_hist", values, step)


def sync_progress_bar(progress_bar, step, **postfix):
    """Advance a tqdm bar to an absolute step count."""
    if progress_bar is None:
        return

    delta = max(0, step - progress_bar.n)
    if delta:
        progress_bar.update(delta)

    if postfix:
        progress_bar.set_postfix(postfix)


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


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    normal_r_mode = bool(getattr(args.method, "normal_r", False))

    if args.method.loss == "dice" and not args.offline:
        raise ValueError("method.loss=dice is only supported when offline=True")
    if normal_r_mode and (not args.offline or args.method.loss != "dice"):
        raise ValueError("method.normal_r=True is only supported when offline=True and method.loss=dice")

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
    pure_cql_offline = bool(getattr(args.method, "offline_pure_cql", False))

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
    force_cql_dice_critic = (
        args.offline
        and args.method.loss == "dice"
        and bool(getattr(args.method, "dice_use_cql_q", False))
        and isinstance(agent, CQL)
    )
    pure_cql_offline = pure_cql_offline or force_cql_dice_critic

    if args.offline:
        train_mode = "offline_pure_cql" if pure_cql_offline else "offline_iq"
    else:
        train_mode = "online_iq"
    print(f"[train_mode] {train_mode}")
    if force_cql_dice_critic:
        print("[dice_mode] method.dice_use_cql_q=True: forcing CQL critic training backend.")

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

    # Track sampled trajectory ids per dataset basename so repeated loads from
    # the same source file (for example experts/ and supplement/ copies) can be
    # kept disjoint within a run.
    sampled_traj_indices = {}

    # Load expert data
    expert_demo_name = os.path.basename(args.env.demo)
    expert_demo_path = hydra.utils.to_absolute_path(f'experts/{expert_demo_name}')
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_indices = expert_memory_replay.load(
        expert_demo_path,
        num_trajs=args.expert.demos,
        sample_freq=args.expert.subsample_freq,
        seed=args.seed + 42,
        return_indices=True)
    sampled_traj_indices[expert_demo_name] = set(expert_indices or [])
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

    # Load offline / supplementary data when running in offline mode
    if args.offline:
        # Start the offline buffer from the already-sampled expert buffer so
        # expert_memory_replay and online_memory_replay share the same expert
        # trajectories rather than independently resampling expert data.
        for transition in expert_memory_replay.buffer:
            online_memory_replay.add(transition)

        # Also mix the current environment's supplementary dataset(s) under
        # iq_learn/supplement into the offline buffer.
        supplement_dir = hydra.utils.to_absolute_path("supplement")
        if os.path.isdir(supplement_dir):
            supplement_files = sorted(
                f for f in os.listdir(supplement_dir)
                if os.path.isfile(os.path.join(supplement_dir, f))
                and f.endswith((".pkl", ".npy", ".pt"))
                and os.path.basename(f) == expert_demo_name)
            if supplement_files:
                print(f'--> Supplement files matched current env: {supplement_files}')
            else:
                print(f'--> No supplement file matched current env demo: {expert_demo_name}')
            for idx, supplement_file in enumerate(supplement_files):
                supplement_name = os.path.basename(supplement_file)
                selected_indices = online_memory_replay.load(
                    os.path.join(supplement_dir, supplement_file),
                    num_trajs=getattr(args.expert, 'offline_demos', -1),
                    sample_freq=args.expert.subsample_freq,
                    seed=args.seed + 44 + idx,
                    exclude_indices=sampled_traj_indices.get(supplement_name),
                    return_indices=True)
                sampled_traj_indices.setdefault(supplement_name, set()).update(
                    selected_indices or [])
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
        if pure_cql_offline and not isinstance(agent, CQL):
            raise ValueError(
                "CQL critic backend requires a CQL agent."
            )

        if not pure_cql_offline:
            agent.iq_update = types.MethodType(iq_update, agent)
            agent.iq_update_critic = types.MethodType(iq_update_critic, agent)

        pbar_desc = "Offline CQL training" if pure_cql_offline else "Offline IQ training"
        print(f"[offline_backend] {'pure_cql_update' if pure_cql_offline else 'iq_update'}")
        if normal_r_mode:
            print("[dice_mode] normal_r=True: directly parameterizing the IQ/DICE reward with the critic architecture")
        offline_eval_episode = 0
        latest_eval_dice = None
        offline_pbar = tqdm(
            range(1, LEARN_STEPS + 1),
            desc=pbar_desc,
            dynamic_ncols=True,
        )

        for learn_step in offline_pbar:
            if pure_cql_offline:
                losses = agent.update(online_memory_replay, logger, learn_step)
            else:
                losses = agent.iq_update(
                    online_memory_replay, expert_memory_replay,
                    logger, learn_step)

            if learn_step % args.log_interval == 0:
                for key, loss in losses.items():
                    writer.add_scalar(key, loss, global_step=learn_step)
                critic_loss = losses.get('critic_loss', losses.get('loss/critic'))
                if critic_loss is not None:
                    offline_pbar.set_postfix(
                        critic=f"{float(critic_loss):.4f}",
                        best=f"{best_eval_returns:.2f}" if np.isfinite(best_eval_returns) else "N/A",
                    )

            if learn_step % int(args.env.eval_interval) == 0:
                if args.method.loss == "dice":
                    eval_dice = _make_dice_agent(agent, args)
                    _log_weight_stats(
                        logger, eval_dice, online_memory_replay, args, learn_step)
                    if _warm_start_eval_bc(eval_dice, latest_eval_dice, args):
                        print("[dice_mode] Warm-started eval BC actor from previous eval.")
                    eval_dice.train_weighted_bc(
                        buffer=online_memory_replay, args=args,
                        logger=logger, writer=writer)
                    latest_eval_dice = eval_dice
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
            dice_agent = _make_dice_agent(agent, args)
            _log_weight_stats(
                logger, dice_agent, online_memory_replay, args, LEARN_STEPS)
            if _warm_start_eval_bc(dice_agent, latest_eval_dice, args):
                print("[dice_mode] Warm-started final BC actor from previous eval.")
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
    obs, next_obs, action, env_reward = batch[0:4]

    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    if bool(getattr(args.method, "normal_r", False)) and args.offline and args.method.loss == "dice":
        # In normal_r mode the critic directly parameterizes the reward r(s,a).
        # Every term in iq_loss only uses (current_Q - gamma*next_v), so the old
        # trick of passing (pred_r + y) as current_Q and next_V as next_v caused
        # a redundant add-then-subtract that is mathematically an identity but
        # introduces catastrophic floating-point cancellation when V is large.
        # Instead, pass pred_r directly as current_Q with a zero next_v so that
        # (current_Q - gamma*0) = pred_r exactly, with no large-number arithmetic.
        zero_next_V = torch.zeros_like(next_V)

        if "DoubleQ" in self.args.q_net._target_:
            pred_r1, pred_r2 = self.critic(obs, action, both=True)
            q1_loss, loss_dict1 = iq_loss(self, pred_r1, current_V, zero_next_V, batch)
            q2_loss, loss_dict2 = iq_loss(self, pred_r2, current_V, zero_next_V, batch)
            critic_loss = 1 / 2 * (q1_loss + q2_loss)
            loss_dict = average_dicts(loss_dict1, loss_dict2)
            loss_dict.update({
                'normal_r/pred_reward_1': pred_r1.mean().item(),
                'normal_r/pred_reward_2': pred_r2.mean().item(),
            })
        else:
            pred_r = self.critic(obs, action)
            critic_loss, loss_dict = iq_loss(self, pred_r, current_V, zero_next_V, batch)
            loss_dict['normal_r/pred_reward'] = pred_r.mean().item()

        loss_dict['critic_loss'] = critic_loss.item()
        loss_dict['loss/critic'] = critic_loss.item()

        logger.log('train/critic_loss', critic_loss, step)
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
        return loss_dict

    agent = self

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
    return loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    # Train transition ensemble on replay buffer (each estimator draws its
    # own independent mini-batch for bootstrap diversity).
    ensemble = getattr(self, "ensemble", None)
    if ensemble is not None:
        ens_losses = ensemble.update_from_buffer(
            policy_buffer, self.batch_size, logger, step)
        losses_ens = {f"train/{k}": v for k, v in ens_losses.items()}
    else:
        losses_ens = {}

    losses = self.iq_update_critic(policy_batch, expert_batch, logger, step)
    losses.update(losses_ens)

    # In normal_r + dice mode the critic parameterizes reward, not Q.  The
    # SAC actor update maximises critic(s,a) as if it were Q, which is
    # semantically wrong (single-step reward ≠ expected return).  Moreover
    # the actor is never used for evaluation (weighted BC replaces it), so
    # updating it only causes out-of-distribution exploitation that can
    # destabilise V estimates used elsewhere.  Skip actor updates entirely.
    _skip_actor = (
        self.args.method.loss == "dice"
        and bool(getattr(self.args.method, "normal_r", False))
    )

    if self.actor and step % self.actor_update_frequency == 0 and not _skip_actor:
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

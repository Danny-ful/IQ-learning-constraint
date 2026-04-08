from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


ENV_SPECS: Dict[str, Dict[str, str]] = {
    "hopper": {
        "config_name": "hopper",
        "gym_env": "Hopper-v2",
        "d4rl_prefix": "hopper",
        "demo_filename": "Hopper-v2_25.pkl",
    },
    "walker": {
        "config_name": "walker",
        "gym_env": "Walker2d-v2",
        "d4rl_prefix": "walker2d",
        "demo_filename": "Walker2d-v2_25.pkl",
    },
    "halfcheetah": {
        "config_name": "cheetah",
        "gym_env": "HalfCheetah-v2",
        "d4rl_prefix": "halfcheetah",
        "demo_filename": "HalfCheetah-v2_25.pkl",
    },
}

QUALITY_CHOICES = ("expert", "medium", "medium-replay", "replay")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert D4RL datasets into this repo's trajectory format and "
            "write matched files into experts/ and supplement/."
        )
    )
    parser.add_argument(
        "--env",
        choices=tuple(ENV_SPECS.keys()) + ("all",),
        default="all",
        help="Environment preset to export.",
    )
    parser.add_argument(
        "--expert-source",
        choices=QUALITY_CHOICES,
        default="expert",
        help="D4RL quality used for experts/.",
    )
    parser.add_argument(
        "--supplement-source",
        choices=QUALITY_CHOICES,
        default="medium",
        help="D4RL quality used for supplement/.",
    )
    parser.add_argument(
        "--expert-episodes",
        type=int,
        default=25,
        help="Maximum number of expert trajectories to save. Use -1 for all.",
    )
    parser.add_argument(
        "--supplement-episodes",
        type=int,
        default=-1,
        help="Maximum number of supplementary trajectories to save. Use -1 for all.",
    )
    parser.add_argument(
        "--skip-expert",
        action="store_true",
        help="Do not export experts/.",
    )
    parser.add_argument(
        "--skip-supplement",
        action="store_true",
        help="Do not export supplement/.",
    )
    return parser


def import_d4rl_modules():
    try:
        import gym  # type: ignore
        import d4rl  # type: ignore  # noqa: F401
    except ImportError as err:
        raise SystemExit(
            "Failed to import gym/d4rl. Please activate the training environment "
            "and install D4RL before running this script."
        ) from err
    return gym


def split_dataset_into_trajectories(dataset: Dict[str, np.ndarray]) -> Dict[str, list]:
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"].astype(bool)
    timeouts = dataset.get("timeouts")
    if timeouts is None:
        timeouts = np.zeros_like(terminals, dtype=bool)
    else:
        timeouts = timeouts.astype(bool)

    next_observations = dataset.get("next_observations")
    if next_observations is None:
        next_observations = np.concatenate([observations[1:], observations[-1:]], axis=0)

    trajs = defaultdict(list)
    current = {
        "states": [],
        "next_states": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    for idx in range(len(observations)):
        current["states"].append(observations[idx])
        current["next_states"].append(next_observations[idx])
        current["actions"].append(actions[idx])
        current["rewards"].append(rewards[idx])
        current["dones"].append(float(terminals[idx]))

        if terminals[idx] or timeouts[idx]:
            trajs["states"].append(np.asarray(current["states"], dtype=np.float32))
            trajs["next_states"].append(np.asarray(current["next_states"], dtype=np.float32))
            trajs["actions"].append(np.asarray(current["actions"], dtype=np.float32))
            trajs["rewards"].append(np.asarray(current["rewards"], dtype=np.float32))
            trajs["dones"].append(np.asarray(current["dones"], dtype=np.float32))
            trajs["lengths"].append(len(current["states"]))
            for key in current:
                current[key].clear()

    if current["states"]:
        trajs["states"].append(np.asarray(current["states"], dtype=np.float32))
        trajs["next_states"].append(np.asarray(current["next_states"], dtype=np.float32))
        trajs["actions"].append(np.asarray(current["actions"], dtype=np.float32))
        trajs["rewards"].append(np.asarray(current["rewards"], dtype=np.float32))
        trajs["dones"].append(np.asarray(current["dones"], dtype=np.float32))
        trajs["lengths"].append(len(current["states"]))

    return trajs


def trim_episodes(trajs: Dict[str, list], max_episodes: int) -> Dict[str, list]:
    if max_episodes is None or max_episodes < 0:
        return trajs

    trimmed = {}
    for key, value in trajs.items():
        trimmed[key] = value[:max_episodes]
    return trimmed


def save_pickle(data: Dict[str, list], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_handle:
        pickle.dump(data, file_handle)


def make_d4rl_name(prefix: str, quality: str) -> str:
    return f"{prefix}-{quality}-v2"


def export_one_dataset(
    gym,
    d4rl_name: str,
    output_path: Path,
    max_episodes: int,
) -> None:
    env = gym.make(d4rl_name)
    dataset = env.get_dataset()
    trajs = split_dataset_into_trajectories(dataset)
    trajs = trim_episodes(trajs, max_episodes)
    save_pickle(trajs, output_path)

    num_episodes = len(trajs["lengths"])
    total_steps = int(sum(trajs["lengths"]))
    avg_length = float(np.mean(trajs["lengths"])) if trajs["lengths"] else 0.0
    print(f"Saved {d4rl_name} -> {output_path}")
    print(f"  episodes={num_episodes}, total_steps={total_steps}, avg_len={avg_length:.1f}")


def iter_envs(env_name: str) -> Iterable[str]:
    if env_name == "all":
        return ENV_SPECS.keys()
    return (env_name,)


def print_train_commands(env_key: str, expert_episodes: int, supplement_episodes: int) -> None:
    spec = ENV_SPECS[env_key]
    print("")
    print(f"[train:{env_key}]")
    print("cd /home/ubuntu/laiwenqi/projects/IQ-learning-constraint/iq_learn")
    print("python train_iq.py \\")
    print(f"  env={spec['config_name']} \\")
    print("  agent=sac \\")
    print("  offline=True \\")
    print("  method.loss=dice \\")
    print(f"  expert.demos={expert_episodes} \\")
    print(f"  expert.offline_demos={supplement_episodes} \\")
    print("  seed=0")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    gym = import_d4rl_modules()

    repo_root = Path(__file__).resolve().parent.parent
    experts_dir = repo_root / "experts"
    supplement_dir = repo_root / "supplement"

    for env_key in iter_envs(args.env):
        spec = ENV_SPECS[env_key]
        demo_filename = spec["demo_filename"]
        d4rl_prefix = spec["d4rl_prefix"]

        print("")
        print(f"=== {env_key} ===")

        if not args.skip_expert:
            export_one_dataset(
                gym=gym,
                d4rl_name=make_d4rl_name(d4rl_prefix, args.expert_source),
                output_path=experts_dir / demo_filename,
                max_episodes=args.expert_episodes,
            )

        if not args.skip_supplement:
            export_one_dataset(
                gym=gym,
                d4rl_name=make_d4rl_name(d4rl_prefix, args.supplement_source),
                output_path=supplement_dir / demo_filename,
                max_episodes=args.supplement_episodes,
            )

        print_train_commands(
            env_key,
            expert_episodes=min(args.expert_episodes, 10) if args.expert_episodes > 0 else 10,
            supplement_episodes=args.supplement_episodes,
        )


if __name__ == "__main__":
    main()

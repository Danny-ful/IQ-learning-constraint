import gym
from agent.sac import SAC
from agent.cql import CQL
from agent.softq import SoftQ
from agent.maxq import MaxQ


def make_agent(env, args):
    obs_dim = env.observation_space.shape[0]

    if args.agent.name == 'maxq':
        print('--> Using MaxQ agent')
        action_dim = env.action_space.n
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = MaxQ(obs_dim, action_dim, args.train.batch, args)
    elif args.agent.name == 'cql':
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise ValueError("agent=cql currently supports continuous action spaces only.")
        print('--> Using CQL agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = CQL(obs_dim, action_dim, action_range, args.train.batch, args)
    elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
        print('--> Using Soft-Q agent')
        action_dim = env.action_space.n
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = SoftQ(obs_dim, action_dim, args.train.batch, args)
    else:
        print('--> Using SAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = SAC(obs_dim, action_dim, action_range, args.train.batch, args)

    return agent

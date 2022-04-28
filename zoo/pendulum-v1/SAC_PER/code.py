import os

from rl import utils
from rl.agents import SAC
from rl.parameters import StepDecay


if __name__ == '__main__':
    utils.set_random_seed(42)

    # achieves > -97 reward at episode 920
    agent = SAC(env='Pendulum-v1', name='sac_per-pendulum', memory_size=50_000,
                actor_lr=StepDecay(1e-4, steps=500, rate=0.1),
                critic_lr=StepDecay(1e-4, steps=500, rate=0.1),
                gamma=0.995, polyak=0.95,
                actor=dict(units=64, num_layers=4, noisy=True),
                critic=dict(units=128),
                use_summary=True, prioritized=True,
                batch_size=256, entropy=1e-3, seed=utils.GLOBAL_SEED)

    # load pretrained agent
    agent.load(path=os.path.join('zoo', 'pendulum-v1', 'SAC_PER', 'weights'))

    # OR train from scratch; T = 200 
    # agent.learn(episodes=1000, timesteps=agent.env._max_episode_steps, save=True,
    #             render=False, evaluation=dict(freq=10, episodes=20),
    #             exploration_steps=10_000)


import os

from rl import utils
from rl.agents import SAC


if __name__ == '__main__':
    utils.set_random_seed(42)

    # about 95 reward at episode 150
    agent = SAC(env='MountainCarContinuous-v0', name='sac-mountain_car', memory_size=100_000,
                polyak=0.99, actor=dict(units=64, num_layers=4, noisy=True),
                critic=dict(units=128), use_summary=True,
                batch_size=256, entropy=1.0, seed=utils.GLOBAL_SEED)

    # load pretrained agent
    agent.load(path=os.path.join('zoo', 'mountain-car-continuous-v0', 'SAC', 'weights'))

    # OR train from scratch
    # agent.learn(episodes=500, timesteps=agent.env._max_episode_steps, save=True,
    #             render=False, evaluation=dict(freq=10, episodes=20),
    #             exploration_steps=10_000)

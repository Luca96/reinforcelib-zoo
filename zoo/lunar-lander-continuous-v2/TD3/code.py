import os

from rl import utils
from rl.agents import TD3


if __name__ == '__main__':
    utils.set_random_seed(42)

    agent = TD3(env='LunarLanderContinuous-v2', actor_lr=1e-3, critic_lr=1e-3, polyak=0.995, actor_update_freq=2,
                memory_size=256_000, batch_size=128, name='td3-lunar', use_summary=True,
                actor=dict(units=64), critic=dict(units=64), noise=0.1, seed=utils.GLOBAL_SEED)

    # load pretrained agent
    agent.load(path=os.path.join('zoo', 'lunar-lander-continuous', 'TD3', 'weights'))

    # OR train from scratch
    # agent.learn(episodes=250, timesteps=200, evaluation=dict(freq=10, episodes=20),
    #             exploration_steps=5 * agent.batch_size, save=True)

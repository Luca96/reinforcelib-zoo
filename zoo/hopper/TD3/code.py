import os
import pybullet_envs

from rl import utils
from rl.agents import TD3


if __name__ == '__main__':
    utils.set_random_seed(42)

    # 2300+ reward at episode 4690
    agent = TD3(env='HopperBulletEnv-v0', actor_lr=1e-3, critic_lr=1e-3, polyak=0.999, actor_update_freq=2,
                memory_size=256_000, batch_size=256, name='td3-hopper', use_summary=True, seed=42,
                actor=dict(units=256), critic=dict(units=256), noise=0.1)

    # load pretrained agent
    agent.load(path=os.path.join('zoo', 'hopper', 'TD3', 'weights'))

    # OR train from scratch
    # agent.learn(episodes=5_000, timesteps=1000, evaluation=dict(freq=20, episodes=20),
    #             save=True, exploration_steps=5 * agent.batch_size)

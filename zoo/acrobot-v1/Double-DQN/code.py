import os

from rl import utils
from rl.agents import DQN
from rl.parameters import StepDecay


if __name__ == '__main__':
    utils.set_random_seed(42)

    # about -76 reward at episode 240
    agent = DQN(env='Acrobot-v1', batch_size=128, policy='e-greedy', clip_norm=None,
                epsilon=StepDecay(0.2, steps=100, rate=0.5), lr=1e-3, name='ddqn-acrobot',
                dueling=False, prioritized=False, double=True, memory_size=50_000,
                gamma=0.99, update_target_network=500, seed=42, use_summary=True,
                network=dict(units=64))

    # load pretrained agent
    agent.load(path=os.path.join('zoo', 'acrobot-v1', 'Double-DQN', 'weights'))

    # OR train from scratch
    # agent.learn(episodes=500, timesteps=agent.env._max_episode_steps, save=True, render=False,
    #             evaluation=dict(episodes=20, freq=10), exploration_steps=512)

import os

from rl import utils
from rl.agents import DQN
from rl.parameters import StepDecay
from rl.layers.preprocessing import MinMaxScaling
from rl.presets import Preset


if __name__ == '__main__':
    utils.set_random_seed(42)

    min_max_scaler = MinMaxScaling(min_value=Preset.CARTPOLE_MIN,
                                   max_value=Preset.CARTPOLE_MAX)

    # DQN solved at episode 90
    agent = DQN(env='CartPole-v1', batch_size=128, policy='e-greedy', clip_norm=None,
                epsilon=StepDecay(0.2, steps=100, rate=0.5), lr=1e-3, name='dqn-cartpole',
                dueling=False, prioritized=False, double=False, memory_size=50_000,
                gamma=0.99, update_target_network=500, seed=42,
                network=dict(units=64, preprocess=dict(state=min_max_scaler)))

    # load pretrained agent
    agent.load(path=os.path.join('zoo', 'cartpole', 'DQN', 'weights'))

    # OR train from scratch
    # agent.learn(episodes=200, timesteps=500, save=True, render=False,
    #             evaluation=dict(episodes=20, freq=10), exploration_steps=512)

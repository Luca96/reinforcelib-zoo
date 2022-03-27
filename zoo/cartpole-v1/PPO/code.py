import os

from rl.agents import PPO
from rl.environments.gym.parallel import SequentialEnv
from rl import utils


if __name__ == '__main__':
    utils.set_random_seed(42)

    # solved at epoch 50
    agent = PPO(env='CartPole-v1', name='ppo-cartpole', horizon=64, batch_size=256,
                optimization_epochs=10, policy_lr=3e-4, num_actors=16,
                entropy=1e-3, clip_norm=(5.0, 5.0), use_summary=True,
                policy=dict(units=32), value=dict(units=64),
                target_kl=0.3, target_mse=1.0, parallel_env=SequentialEnv,
                seed=utils.GLOBAL_SEED)

    agent.load(os.path.join('zoo', 'cartpole', 'PPO', 'weights'))

    # or train from scratch
    # agent.learn(episodes=100, timesteps=500, save=True, should_close=True,
    #             evaluation=dict(episodes=25, freq=10))

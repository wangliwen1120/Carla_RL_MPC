from gym.envs.registration import register

register(
        id='gym_env-v0',
        entry_point='Carla_gym.envs:CarlagymEnv',
)
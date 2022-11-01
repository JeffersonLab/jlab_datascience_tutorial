from gym.envs.registration import register, make

register(
    id='Surrogate_Accelerator-v1',
    entry_point='rl_tutorial.envs.rl_envs:Surrogate_Accelerator_v1'
)


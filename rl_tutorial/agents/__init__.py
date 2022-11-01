from rl_tutorial.agents.registration import register, make

register(
    id='KerasDQN-v1',
    entry_point='rl_tutorial.agents.rl_agents:DDQN_Agent'
)
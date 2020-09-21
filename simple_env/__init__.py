from gym.envs.registration import register

register(
    id='all_ones-v0',
    entry_point='simple_env.envs:AllOnes',
    timestep_limit=1024,
    nondeterministic=True,
)
register(
    id='copy_cat-v0',
    entry_point='simple_env.envs:CopyCat',
    timestep_limit=1024,
    nondeterministic=True,
)

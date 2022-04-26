import gym

gym.register("MineWorldEnv-v0",
             entry_point="automaton_transfer.lib.env.mineworldenv:MineWorldEnv",
             nondeterministic=False,
             max_episode_steps=1000)

gym.register("ObtainDiamondGridworld-v0",
             entry_point="automaton_transfer.lib.env.diamondmineworldenv:ObtainDiamondGridworld",
             nondeterministic=False,
             max_episode_steps=10000
             )

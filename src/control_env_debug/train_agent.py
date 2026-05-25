from .rl_agents.TD3.td3 import TD3Agent


def make_agent(env, **overrides):
    """Create TD3 agent with optional overrides and a no-RM baseline.

    Recognized overrides:
        - actor_lr, critic_lr, batch_size, policy_noise, noise_clip, gamma, tau,
            buffer_size, policy_delay, use_crm
        - no_rm: if True, build networks with num_rm_states=1 (ignore RM state)
    """
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = (env.action_space.low, env.action_space.high)
    no_rm = bool(overrides.get('no_rm', False))
    num_rm_states = 1 if no_rm else env.automaton.num_states
    agent = TD3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_bounds=action_bounds,
        num_rm_states=int(num_rm_states),
        use_crm=bool(overrides.get('use_crm', False)),
        actor_lr=overrides.get('actor_lr', 1e-3),
        critic_lr=overrides.get('critic_lr', 1e-3),
        batch_size=int(overrides.get('batch_size', 100)),
        policy_noise=overrides.get('policy_noise', 0.5),
        noise_clip=overrides.get('noise_clip', 0.5),
        gamma=overrides.get('gamma', 0.99),
        tau=overrides.get('tau', 0.005),
        buffer_size=int(overrides.get('buffer_size', 1e6)),
        policy_delay=int(overrides.get('policy_delay', 2)),
        device=overrides.get('device', None),
    )
    return agent

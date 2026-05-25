import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utils.replay_buffer import ReplayBuffer


class _AutomatonTransferHelper:
    """Implements teacher Qavg and beta weights β(ω,σ)=ρ^{η_student(ω,σ)} with gating on teacher counts.

    The teacher Q-automaton provides abstract Q-values over RM-state transitions.
    These are mixed into the student's TD target as a **non-negative additive bonus**:

        target_q_mixed = target_q + β * (Qavg(u,σ) - Qmin(u))

    The bonus ``Qavg - Qmin`` is always ≥ 0: the *worst* transition from a
    state gets bonus 0 while better transitions get a positive bonus
    proportional to their relative quality.  This avoids the systematic
    negative bias that the old ``Qavg - Vavg`` form (always ≤ 0) introduced.

    When ``static_weight`` is provided, β is held constant (gated by teacher
    counts) instead of decaying with student experience.
    """
    def __init__(self, *, source_q_total, source_q_count, num_states: int, num_aps: int,
                 rho: float = 0.999, min_source_q_count: int = 1, device: torch.device,
                 eta_div: float = 256.0, beta_min: float = 0.0,
                 static_weight: float = None, initial_weight: float = 1.0):
        self.device = device
        self.num_states = int(num_states)
        self.num_aps = int(num_aps)
        self.rho = float(rho)
        self.min_source_q_count = int(max(1, min_source_q_count))
        self.eta_div = float(max(1.0, eta_div))
        self.beta_min = float(max(0.0, beta_min))
        self.static_weight = static_weight  # None → dynamic, float → fixed β
        self.initial_weight = float(initial_weight)  # scales dynamic β
        # Tensors for fast gather
        self.source_q_total = torch.as_tensor(source_q_total, dtype=torch.float32, device=device)
        self.source_q_count = torch.as_tensor(source_q_count, dtype=torch.int64, device=device)
        # Cached averages with safe division
        self._cached_qavg = torch.where(self.source_q_count.ne(0),
                                        self.source_q_total / self.source_q_count.clamp_min(1).to(torch.float32),
                                        torch.zeros_like(self.source_q_total, dtype=torch.float32))
        # Compute per-state Q bounds across valid actions
        has_count = self.source_q_count >= self.min_source_q_count
        masked_q_hi = torch.where(has_count, self._cached_qavg, torch.full_like(self._cached_qavg, -1e9))
        masked_q_lo = torch.where(has_count, self._cached_qavg, torch.full_like(self._cached_qavg, 1e9))
        self._cached_vavg = masked_q_hi.max(dim=1).values  # V = max_a Q  [num_states]
        self._cached_qmin = masked_q_lo.min(dim=1).values   # min_a Q     [num_states]
        # Fallback: if no action is valid for a state, V=0, Qmin=0
        any_valid = has_count.any(dim=1)
        self._cached_vavg = torch.where(any_valid, self._cached_vavg, torch.zeros(num_states, device=device))
        self._cached_qmin = torch.where(any_valid, self._cached_qmin, torch.zeros(num_states, device=device))
        # Student counts on CPU (updated per sampled batch)
        self.target_q_count = torch.zeros((self.num_states, self.num_aps), dtype=torch.int64)

    def teacher_advantage(self, u_prev: torch.Tensor, ap_idx: torch.Tensor) -> torch.Tensor:
        """Non-negative transition quality: Q(u,σ) - Q_min(u) ≥ 0.

        Returns 0 for the worst transition from state u and a positive
        value for better transitions.
        """
        q = self._cached_qavg[u_prev, ap_idx]
        q_min = self._cached_qmin[u_prev]
        return (q - q_min).clamp(min=0.0)

    def teacher_qavg(self, u_prev: torch.Tensor, ap_idx: torch.Tensor) -> torch.Tensor:
        # Returns vector [B] of Qavg_teacher(ω,σ)
        return self._cached_qavg[u_prev, ap_idx]

    def beta_weights(self, u_prev: torch.Tensor, ap_idx: torch.Tensor) -> torch.Tensor:
        # Gate: only apply mixing if teacher saw transition at least min_source_q_count
        gate = (self.source_q_count[u_prev, ap_idx] >= self.min_source_q_count)
        if self.static_weight is not None:
            # Static distillation: constant β
            beta = torch.full((u_prev.shape[0],), self.static_weight,
                              dtype=torch.float32, device=self.device)
        else:
            # Dynamic distillation: β = initial_weight * ρ^{η_student(ω,σ)/η_div}
            eta_vals = self.target_q_count[u_prev.cpu().numpy(), ap_idx.cpu().numpy()]
            eta_t = torch.as_tensor(eta_vals, dtype=torch.float32, device=self.device) / self.eta_div
            beta = self.initial_weight * torch.pow(torch.tensor(self.rho, dtype=torch.float32, device=self.device), eta_t)
            if self.beta_min > 0.0:
                beta = torch.clamp(beta, min=self.beta_min)
        beta = torch.where(gate, beta, torch.zeros_like(beta))
        return beta

    def update_counts(self, u_prev: np.ndarray, ap_idx: np.ndarray):
        # Increment counts for unique (u, ap) in this sampled batch
        if u_prev.size == 0:
            return
        valid_mask = ap_idx >= 0
        if not np.any(valid_mask):
            return
        u_v = u_prev[valid_mask]
        a_v = ap_idx[valid_mask]
        # Use np.add.at for scatter-add (no Python loop)
        np.add.at(self.target_q_count.numpy(), (u_v, a_v), 1)


class Actor(nn.Module):
    """Actor network: outputs actions given state and RM one-hot state."""
    def __init__(self, obs_dim, action_dim, max_action, num_rm_states):
        super().__init__()
        self.max_action = float(max_action)
        self.l1 = nn.Linear(obs_dim + num_rm_states, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state, rm_state_onehot):
        x = torch.cat([state, rm_state_onehot], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    """Critic network: outputs Q-value for (state, rm, action)."""
    def __init__(self, obs_dim, action_dim, num_rm_states):
        super().__init__()
        self.l1 = nn.Linear(obs_dim + num_rm_states + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, rm_state_onehot, action):
        x = torch.cat([state, rm_state_onehot, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class TD3Agent:
    """TD3 agent with optional Counterfactual Replay (CRM)."""
    def __init__(self, obs_dim, action_dim, action_bounds, num_rm_states, **kwargs):
        device_str = kwargs.get('device', None)
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_crm = kwargs.get('use_crm', False)

        obs_dim_only = obs_dim
        # Use bound magnitude as scalar max (assumes symmetric bounds)
        max_action = float(np.max(np.abs(action_bounds[1])))

        # Networks
        self.actor = Actor(obs_dim_only, action_dim, max_action, num_rm_states).to(self.device)
        self.actor_target = Actor(obs_dim_only, action_dim, max_action, num_rm_states).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=kwargs.get('actor_lr', 1e-3))

        self.critic_1 = Critic(obs_dim_only, action_dim, num_rm_states).to(self.device)
        self.critic_1_target = Critic(obs_dim_only, action_dim, num_rm_states).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=kwargs.get('critic_lr', 1e-3))

        self.critic_2 = Critic(obs_dim_only, action_dim, num_rm_states).to(self.device)
        self.critic_2_target = Critic(obs_dim_only, action_dim, num_rm_states).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=kwargs.get('critic_lr', 1e-3))

        # Replay buffer and scalars
        self.buffer = ReplayBuffer(kwargs.get('buffer_size', 1e6))
        self.num_rm_states = int(num_rm_states)
        self.max_action = max_action

        # Hyperparameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)
        self.batch_size = int(kwargs.get('batch_size', 256))
        self.policy_delay = int(kwargs.get('policy_delay', 2))
        self.policy_noise = kwargs.get('policy_noise', 0.2)
        self.noise_clip = kwargs.get('noise_clip', 0.5)
        self.train_step_counter = 0

        # CRM diagnostics
        self.crm_enabled = bool(self.use_crm)
        self.crm_real_steps = 0
        self.crm_synth_steps = 0
        self.crm_synth_last = 0

        # Automaton adjacency and dynamic target mixing helper (None when disabled)
        self._adj = None
        self._aut_transfer = None

    def enable_target_mixing(self, *, source_q_total, source_q_count, adj: np.ndarray,
                              rho: float = 0.999, min_source_q_count: int = 1,
                              eta_div: float = 256.0, beta_min: float = 0.0,
                              static_weight: float = None, initial_weight: float = 1.0):
        """Enable automaton transfer: target mixing with teacher Qavg and β weights.

        adj: [num_states, num_aps] mapping (u, ap) -> next_u
        static_weight: if provided, use a constant β instead of ρ^η annealing.
        """
        self._adj = np.asarray(adj, dtype=np.int64)
        num_aps = int(self._adj.shape[1])
        # Precompute reverse lookup: (u_prev, u_next) -> ap_idx for vectorized inference
        num_states = self._adj.shape[0]
        # _adj_reverse[u_prev, u_next] = ap_idx (or -1 if no such transition)
        self._adj_reverse = np.full((num_states, num_states), -1, dtype=np.int64)
        for u in range(num_states):
            for ap in range(num_aps):
                u_next = self._adj[u, ap]
                if 0 <= u_next < num_states:
                    self._adj_reverse[u, u_next] = ap
        self._aut_transfer = _AutomatonTransferHelper(
            source_q_total=source_q_total,
            source_q_count=source_q_count,
            num_states=self.num_rm_states,
            num_aps=num_aps,
            rho=rho,
            min_source_q_count=min_source_q_count,
            device=self.device,
            eta_div=eta_div,
            beta_min=beta_min,
            static_weight=static_weight,
            initial_weight=initial_weight,
        )

    def get_action(self, obs, rm_state, add_noise=True):
        # When running a no-RM baseline, force RM state to 0 to match num_rm_states=1
        if getattr(self, 'num_rm_states', 1) == 1:
            rm_state = 0
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        rm_state_tensor = F.one_hot(torch.tensor([rm_state], device=self.device), self.num_rm_states).float()
        action = self.actor(obs_tensor, rm_state_tensor).cpu().data.numpy().flatten()
        if add_noise:
            noise_scale = self.max_action * (2.0 if rm_state == 2 else 0.5)
            action = action + np.random.normal(0, noise_scale, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def get_actions_batch(self, obs_batch: np.ndarray, rm_states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Batched action selection — single forward pass for all envs."""
        n = obs_batch.shape[0]
        if getattr(self, 'num_rm_states', 1) == 1:
            rm_states = np.zeros(n, dtype=int)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            rm_t = F.one_hot(torch.as_tensor(rm_states.astype(int), dtype=torch.long, device=self.device),
                             self.num_rm_states).float()
            actions = self.actor(obs_t, rm_t).cpu().numpy()
        if add_noise:
            noise_scales = np.where(rm_states == 2, self.max_action * 2.0, self.max_action * 0.5)
            actions = actions + np.random.normal(0, 1, size=actions.shape) * noise_scales[:, None]
        return np.clip(actions, -self.max_action, self.max_action)

    def store_experience(self, obs, rm_state, action, reward, next_obs, next_rm_state, done, info):
        if self.use_crm:
            exps = info.get('crm_experiences', [])
            # Count diagnostics
            self.crm_real_steps += 1
            self.crm_synth_steps += len(exps)
            self.crm_synth_last = len(exps)
            for exp in exps:
                self.buffer.add(exp)
            # Also add the real transition
            self.buffer.add((obs, rm_state, action, reward, next_obs, next_rm_state, done))
        else:
            self.buffer.add((obs, rm_state, action, reward, next_obs, next_rm_state, done))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None
        self.train_step_counter += 1

        batch = self.buffer.sample_arrays(self.batch_size)
        obs_b, u_b, action_b, reward_b, next_obs_b, next_u_b, done_b = batch

        # Clamp RM states to 0 if running without RM (num_rm_states==1)
        if getattr(self, 'num_rm_states', 1) == 1:
            u_b = np.zeros_like(u_b, dtype=int)
            next_u_b = np.zeros_like(next_u_b, dtype=int)

        state = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)
        rm_state = F.one_hot(
            torch.as_tensor(u_b.astype(int).flatten(), dtype=torch.long, device=self.device),
            self.num_rm_states
        ).float()
        action = torch.as_tensor(action_b, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward_b.reshape(-1, 1), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_obs_b, dtype=torch.float32, device=self.device)
        next_rm_state = F.one_hot(
            torch.as_tensor(next_u_b.astype(int).flatten(), dtype=torch.long, device=self.device),
            self.num_rm_states
        ).float()
        done = torch.as_tensor(done_b.reshape(-1, 1), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_rm_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1 = self.critic_1_target(next_state, next_rm_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_rm_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.gamma * target_q
            mix_valid_frac = None
            mix_beta_mean = None
            mix_tq_mean = None
            mix_td_mean = float(target_q.mean().item()) if target_q.numel() > 0 else 0.0
            # Dynamic automaton target mixing (teacher advantage + β)
            if self._aut_transfer is not None and self._adj is not None:
                try:
                    # Vectorized AP inference using precomputed reverse lookup
                    u_prev_np = u_b.astype(int).ravel()
                    u_next_np = next_u_b.astype(int).ravel()
                    ap_idx = self._adj_reverse[u_prev_np, u_next_np]  # [B], -1 for invalid
                    # Prepare tensors and mask
                    ap_idx_t = torch.as_tensor(ap_idx, dtype=torch.long, device=self.device)
                    valid = ap_idx_t.ge(0)
                    if torch.any(valid):
                        u_prev_t = torch.as_tensor(u_prev_np, dtype=torch.long, device=self.device)
                        # Additive advantage bonus: target += β * A_teacher(u,σ)
                        adv = self._aut_transfer.teacher_advantage(u_prev_t, ap_idx_t).unsqueeze(1)
                        tq = self._aut_transfer.teacher_qavg(u_prev_t, ap_idx_t).unsqueeze(1)
                        beta = self._aut_transfer.beta_weights(u_prev_t, ap_idx_t).unsqueeze(1)
                        target_q = torch.where(valid.view(-1, 1), target_q + beta * adv, target_q)
                        # Metrics
                        vmask = valid.view(-1, 1)
                        mix_valid_frac = float(valid.float().mean().item())
                        try:
                            mix_beta_mean = float(beta[vmask].mean().item())
                        except Exception:
                            mix_beta_mean = 0.0
                        try:
                            mix_tq_mean = float(tq[vmask].mean().item())
                        except Exception:
                            mix_tq_mean = 0.0
                    # Update student counts from this sampled batch
                    self._aut_transfer.update_counts(u_prev_np, ap_idx)
                except Exception:
                    pass
        # Critic updates
        current_q1 = self.critic_1(state, rm_state, action)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        current_q2 = self.critic_2(state, rm_state, action)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Delayed actor + target updates
        if self.train_step_counter % self.policy_delay == 0:
            actor_loss = -self.critic_1(state, rm_state, self.actor(state, rm_state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft updates
            with torch.no_grad():
                for p, tp in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, tp in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            'critic1': float(critic_1_loss.item()),
            'critic2': float(critic_2_loss.item()),
            'crm_last_synth': int(self.crm_synth_last) if self.crm_enabled else 0,
            'crm_total_real': int(self.crm_real_steps) if self.crm_enabled else 0,
            'crm_total_synth': int(self.crm_synth_steps) if self.crm_enabled else 0,
            'mix_valid_frac': 0.0 if mix_valid_frac is None else mix_valid_frac,
            'mix_beta_mean': 0.0 if mix_beta_mean is None else mix_beta_mean,
            'mix_tq_mean': 0.0 if mix_tq_mean is None else mix_tq_mean,
            'mix_td_mean': mix_td_mean,
        }

    # Convenience aliases expected by some utilities
    def act(self, obs):
        return self.get_action(obs, rm_state=0, add_noise=False)

    def store(self, obs, rm_state, action, reward, next_obs, next_rm_state, done):
        self.store_experience(obs, rm_state, action, reward, next_obs, next_rm_state, done, info={})

    def save(self, filename):
        torch.save(self.critic_1.state_dict(), filename + "_critic_1.pth")
        torch.save(self.critic_2.state_dict(), filename + "_critic_2.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")

    def load(self, filename):
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1.pth", map_location=self.device))
        self.critic_2.load_state_dict(torch.load(filename + "_critic_2.pth", map_location=self.device))
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=self.device))
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

    def load_compatible(self, filename):
        """Load teacher weights with best-effort shape matching (C-PREP style).

        If all shapes match, this is identical to ``load()``.
        If some layers have mismatched shapes (e.g. different obs_dim due to
        teacher/student env differences), those layers are skipped and a
        warning is printed.  Returns the number of layers that were
        successfully transferred.
        """
        transferred = 0
        skipped = 0
        for net_name, net in [("actor", self.actor),
                              ("critic_1", self.critic_1),
                              ("critic_2", self.critic_2)]:
            src = torch.load(filename + f"_{net_name}.pth", map_location=self.device)
            tgt = net.state_dict()
            compatible = {}
            for k in tgt:
                if k in src and src[k].shape == tgt[k].shape:
                    compatible[k] = src[k]
                    transferred += 1
                else:
                    skipped += 1
                    shape_src = tuple(src[k].shape) if k in src else "MISSING"
                    print(f"  [load_compatible] skip {net_name}.{k}: "
                          f"src={shape_src} vs tgt={tuple(tgt[k].shape)}")
            net.load_state_dict(compatible, strict=False)
        # Sync targets
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
        print(f"  [load_compatible] transferred {transferred} params, "
              f"skipped {skipped}")
        return transferred

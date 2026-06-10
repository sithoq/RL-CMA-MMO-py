"""Multi-preference shared Double DQN.

The agent keeps a shared state encoder and several Q-heads. All heads use the
same discrete action space, but each head scalarizes the stored reward vector
with a different preference weight. This keeps the online data efficient while
allowing coverage-, quality-, diversity-, and balanced-search preferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_HEAD_WEIGHTS: dict[str, tuple[float, ...]] = {
    # Reward vector order:
    # [archive_gain, fitness_improvement, novelty/diversity, replacement_success, local_rank/budget]
    "coverage": (0.45, 0.10, 0.30, 0.10, 0.05),
    "quality": (0.10, 0.55, 0.05, 0.25, 0.05),
    "diversity": (0.25, 0.05, 0.50, 0.10, 0.10),
    "balanced": (0.25, 0.25, 0.20, 0.20, 0.10),
}


class MultiHeadQNetwork(nn.Module):
    """Shared state encoder with one linear Q-head per preference."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, head_names: list[str]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, action_dim) for name in head_names})

    def forward(self, x: torch.Tensor, head_name: str) -> torch.Tensor:
        return self.heads[head_name](self.encoder(x))


class RewardVectorReplayBuffer:
    """Fixed-size circular replay buffer storing reward vectors."""

    def __init__(self, capacity: int, state_dim: int, reward_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.reward_dim = int(reward_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.reward_vectors = np.zeros((self.capacity, self.reward_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward_vector: np.ndarray, next_state: np.ndarray) -> None:
        idx = self.ptr % self.capacity
        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.actions[idx] = int(action)
        reward = np.asarray(reward_vector, dtype=np.float32).reshape(self.reward_dim)
        self.reward_vectors[idx] = np.clip(reward, -5.0, 5.0)
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        idx = rng.choice(self.size, size=int(batch_size), replace=False)
        return self.states[idx], self.actions[idx], self.reward_vectors[idx], self.next_states[idx]


@dataclass
class MultiPreferenceDQNConfig:
    state_dim: int = 16
    action_dim: int = 27
    reward_dim: int = 5
    hidden_dim: int = 64
    buffer_capacity: int = 20000
    batch_size: int = 128
    gamma: float = 0.90
    tau: float = 0.02
    lr: float = 1e-3
    epsilon0: float = 0.10
    temperature0: float = 2.0
    q_clip: float = 10.0
    grad_clip: float = 1.0
    action_credit_enabled: bool = True
    action_credit_alpha: float = 0.10
    action_credit_bonus: float = 1.0
    action_credit_ucb: float = 0.15
    head_weights: dict[str, tuple[float, ...]] = field(default_factory=lambda: dict(DEFAULT_HEAD_WEIGHTS))


class MultiPreferenceDoubleDQNAgent:
    """Shared-encoder Double DQN with multiple preference heads."""

    def __init__(self, config: MultiPreferenceDQNConfig | None = None, device: str | None = None):
        self.config = config or MultiPreferenceDQNConfig()
        self.head_names = list(self.config.head_weights.keys())
        if not self.head_names:
            raise ValueError("At least one preference head is required.")

        for name, weights in self.config.head_weights.items():
            if len(weights) != self.config.reward_dim:
                raise ValueError(f"Head {name!r} weight length must equal reward_dim.")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = MultiHeadQNetwork(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim, self.head_names
        ).to(self.device)
        self.target = MultiHeadQNetwork(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim, self.head_names
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = RewardVectorReplayBuffer(
            self.config.buffer_capacity, self.config.state_dim, self.config.reward_dim
        )
        self.action_hist = np.zeros(self.config.action_dim, dtype=int)
        self.head_hist = np.zeros(len(self.head_names), dtype=int)
        self.action_credit = np.zeros((len(self.head_names), self.config.action_dim), dtype=float)
        self.action_count = np.zeros((len(self.head_names), self.config.action_dim), dtype=float)

    @torch.no_grad()
    def q_values(self, state: np.ndarray, head_name: str) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        return self.online(s, head_name).cpu().numpy().reshape(-1)

    def action_scores(self, state: np.ndarray, head_name: str) -> np.ndarray:
        """Return Q values augmented with fast per-head action credit."""

        q = self.q_values(state, head_name)
        if not self.config.action_credit_enabled:
            return q

        head_idx = self._head_index(head_name)
        credit = self.action_credit[head_idx]
        max_abs = float(np.max(np.abs(credit)))
        credit_norm = credit / max_abs if max_abs > 1e-12 else np.zeros_like(credit)
        total = float(np.sum(self.action_count[head_idx]))
        ucb = np.sqrt(np.log(total + 1.0) / (self.action_count[head_idx] + 1.0))
        return q + self.config.action_credit_bonus * credit_norm + self.config.action_credit_ucb * ucb

    def choose_head(self, progress: float, rng: np.random.Generator) -> str:
        """Progress-only fallback head schedule used by the v1 MPDQN algorithm."""

        progress = float(np.clip(progress, 0.0, 1.0))
        if progress < 0.45:
            probs = {"coverage": 0.45, "diversity": 0.30, "balanced": 0.20, "quality": 0.05}
        elif progress < 0.75:
            probs = {"balanced": 0.50, "coverage": 0.20, "diversity": 0.20, "quality": 0.10}
        else:
            probs = {"balanced": 0.45, "quality": 0.35, "coverage": 0.10, "diversity": 0.10}

        weights = np.asarray([probs.get(name, 0.0) for name in self.head_names], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(self.head_names), dtype=float)
        weights = weights / weights.sum()
        head_idx = int(rng.choice(len(self.head_names), p=weights))
        self.head_hist[head_idx] += 1
        return self.head_names[head_idx]

    def select_action(
        self,
        state: np.ndarray,
        progress: float,
        rng: np.random.Generator,
        head_name: str | None = None,
    ) -> int:
        """Select an action with epsilon + softmax exploration."""

        progress = float(np.clip(progress, 0.0, 1.0))
        head = head_name or self.choose_head(progress, rng)
        if head_name is not None:
            self.head_hist[self.head_names.index(head)] += 1

        epsilon = self.config.epsilon0 * (1.0 - progress)
        if rng.random() < epsilon:
            action = int(rng.integers(self.config.action_dim))
        else:
            q = self.action_scores(state, head)
            temperature = max(0.3, self.config.temperature0 * (1.0 - progress))
            q = q - np.max(q)
            probs = np.exp(q / temperature)
            probs = probs / np.sum(probs)
            action = int(rng.choice(self.config.action_dim, p=probs))
        self.action_hist[action] += 1
        self.action_count[self._head_index(head), action] += 1.0
        return action

    def remember(self, state: np.ndarray, action: int, reward_vector: np.ndarray, next_state: np.ndarray) -> None:
        self.buffer.add(state, action, reward_vector, next_state)

    def update_action_credit(self, head_name: str, action: int, reward_vector: np.ndarray) -> None:
        """Update the fast bandit-style credit for one head/action pair."""

        if not self.config.action_credit_enabled:
            return
        head_idx = self._head_index(head_name)
        action = int(np.clip(action, 0, self.config.action_dim - 1))
        reward = np.asarray(reward_vector, dtype=float).reshape(self.config.reward_dim)
        weights = np.asarray(self.config.head_weights[head_name], dtype=float)
        scalar_reward = float(np.dot(reward, weights))
        alpha = float(np.clip(self.config.action_credit_alpha, 0.0, 1.0))
        old = self.action_credit[head_idx, action]
        self.action_credit[head_idx, action] = (1.0 - alpha) * old + alpha * scalar_reward

    def head_action_credit_top(self, top_k: int = 3) -> str:
        """CSV-friendly top credited actions per head."""

        parts: list[str] = []
        k = max(1, int(top_k))
        for head_idx, head_name in enumerate(self.head_names):
            order = np.argsort(self.action_credit[head_idx])[::-1][:k]
            items = ",".join(f"{int(a)}:{self.action_credit[head_idx, int(a)]:.4g}" for a in order)
            parts.append(f"{head_name}={items}")
        return "|".join(parts)

    def train_step(self, rng: np.random.Generator) -> dict[str, float] | None:
        if self.buffer.size < self.config.batch_size:
            return None

        s, a, rv, ns = self.buffer.sample(self.config.batch_size, rng)
        states = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(a, dtype=torch.long, device=self.device).view(-1, 1)
        reward_vectors = torch.as_tensor(rv, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(ns, dtype=torch.float32, device=self.device)

        total_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        losses: dict[str, float] = {}
        for head_name in self.head_names:
            weights = torch.as_tensor(
                self.config.head_weights[head_name], dtype=torch.float32, device=self.device
            )
            rewards = torch.sum(reward_vectors * weights.view(1, -1), dim=1)
            q_pred = self.online(states, head_name).gather(1, actions).squeeze(1)
            with torch.no_grad():
                next_actions = torch.argmax(self.online(next_states, head_name), dim=1, keepdim=True)
                q_next = self.target(next_states, head_name).gather(1, next_actions).squeeze(1)
                q_target = rewards + self.config.gamma * q_next
                q_target = torch.clamp(q_target, -self.config.q_clip, self.config.q_clip)

            head_loss = F.smooth_l1_loss(q_pred, q_target)
            total_loss = total_loss + head_loss
            losses[head_name] = float(head_loss.detach().cpu().item())

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self._soft_update()
        return losses

    def _soft_update(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_param, online_param in zip(self.target.parameters(), self.online.parameters(), strict=False):
                target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    def _head_index(self, head_name: str) -> int:
        return self.head_names.index(head_name)

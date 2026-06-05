"""共享 Double DQN。

这里不用 Stable-Baselines3，而是保留研究算法可控性：动作空间、reward、target
network 更新和经验回放都显式写出，方便后续做消融实验。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """固定容量循环经验池。"""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        idx = self.ptr % self.capacity
        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(np.clip(reward, -5.0, 5.0))
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        idx = rng.choice(self.size, size=int(batch_size), replace=False)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]


@dataclass
class DQNConfig:
    state_dim: int = 7
    action_dim: int = 27
    hidden_dim: int = 64
    buffer_capacity: int = 5000
    batch_size: int = 64
    gamma: float = 0.90
    tau: float = 0.02
    lr: float = 1e-3
    epsilon0: float = 0.10
    temperature0: float = 2.0
    q_clip: float = 10.0
    grad_clip: float = 1.0


class DoubleDQNAgent:
    """在线网络负责选动作，target 网络负责稳定估值。"""

    def __init__(self, config: DQNConfig | None = None, device: str | None = None):
        self.config = config or DQNConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(self.config.state_dim, self.config.action_dim, self.config.hidden_dim).to(self.device)
        self.target = QNetwork(self.config.state_dim, self.config.action_dim, self.config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_capacity, self.config.state_dim)
        self.action_hist = np.zeros(self.config.action_dim, dtype=int)

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        return self.online(s).cpu().numpy().reshape(-1)

    def select_action(self, state: np.ndarray, progress: float, rng: np.random.Generator) -> int:
        """epsilon + softmax 混合探索，避免早期动作坍缩。"""

        progress = float(np.clip(progress, 0.0, 1.0))
        epsilon = self.config.epsilon0 * (1.0 - progress)
        if rng.random() < epsilon:
            action = int(rng.integers(self.config.action_dim))
        else:
            q = self.q_values(state)
            temperature = max(0.3, self.config.temperature0 * (1.0 - progress))
            q = q - np.max(q)
            probs = np.exp(q / temperature)
            probs = probs / np.sum(probs)
            action = int(rng.choice(self.config.action_dim, p=probs))
        self.action_hist[action] += 1
        return action

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self.buffer.add(state, action, reward, next_state)

    def train_step(self, rng: np.random.Generator) -> float | None:
        if self.buffer.size < self.config.batch_size:
            return None
        s, a, r, ns = self.buffer.sample(self.config.batch_size, rng)
        states = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(a, dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(ns, dtype=torch.float32, device=self.device)

        q_pred = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Double DQN：online 选下一动作，target 评估该动作。
            next_actions = torch.argmax(self.online(next_states), dim=1, keepdim=True)
            q_next = self.target(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + self.config.gamma * q_next
            q_target = torch.clamp(q_target, -self.config.q_clip, self.config.q_clip)

        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self._soft_update()
        return float(loss.detach().cpu().item())

    def _soft_update(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_param, online_param in zip(self.target.parameters(), self.online.parameters(), strict=False):
                target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

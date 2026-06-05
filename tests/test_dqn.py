import numpy as np
import pytest

pytest.importorskip("torch")

from rlmmo.rl.dqn import DQNConfig, DoubleDQNAgent


def test_double_dqn_train_step_runs():
    rng = np.random.default_rng(3)
    cfg = DQNConfig(state_dim=7, action_dim=27, hidden_dim=16, batch_size=4, buffer_capacity=20)
    agent = DoubleDQNAgent(cfg, device="cpu")
    for _ in range(8):
        s = rng.random(7)
        ns = rng.random(7)
        agent.remember(s, int(rng.integers(27)), float(rng.normal()), ns)
    loss = agent.train_step(rng)
    assert loss is not None
    action = agent.select_action(rng.random(7), 0.1, rng)
    assert 0 <= action < 27

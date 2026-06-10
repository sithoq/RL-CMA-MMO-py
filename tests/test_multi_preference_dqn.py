import numpy as np
import pytest

pytest.importorskip("torch")

from rlmmo.rl.multi_preference_dqn import MultiPreferenceDQNConfig, MultiPreferenceDoubleDQNAgent


def test_multi_preference_agent_trains_on_reward_vectors():
    rng = np.random.default_rng(17)
    cfg = MultiPreferenceDQNConfig(
        state_dim=5,
        action_dim=7,
        reward_dim=5,
        hidden_dim=16,
        batch_size=4,
        buffer_capacity=20,
    )
    agent = MultiPreferenceDoubleDQNAgent(cfg, device="cpu")

    for _ in range(8):
        agent.remember(
            state=rng.random(5),
            action=int(rng.integers(7)),
            reward_vector=rng.random(5),
            next_state=rng.random(5),
        )

    losses = agent.train_step(rng)

    assert losses is not None
    assert set(losses) == {"coverage", "quality", "diversity", "balanced"}
    assert all(np.isfinite(v) for v in losses.values())


def test_multi_preference_agent_selects_valid_action_for_each_head():
    rng = np.random.default_rng(19)
    cfg = MultiPreferenceDQNConfig(state_dim=5, action_dim=7, reward_dim=5, hidden_dim=16)
    agent = MultiPreferenceDoubleDQNAgent(cfg, device="cpu")
    state = rng.random(5)

    for head_name in agent.head_names:
        action = agent.select_action(state, progress=0.25, rng=rng, head_name=head_name)
        assert 0 <= action < 7

    assert sum(agent.action_hist) == len(agent.head_names)
    assert sum(agent.head_hist) == len(agent.head_names)

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


def test_action_credit_increases_head_action_score():
    rng = np.random.default_rng(23)
    cfg = MultiPreferenceDQNConfig(
        state_dim=5,
        action_dim=7,
        reward_dim=5,
        hidden_dim=16,
        action_credit_enabled=True,
        action_credit_alpha=1.0,
        action_credit_bonus=3.0,
        action_credit_ucb=0.0,
    )
    agent = MultiPreferenceDoubleDQNAgent(cfg, device="cpu")
    state = rng.random(5)

    before = agent.action_scores(state, "coverage").copy()
    agent.update_action_credit("coverage", 3, np.array([1.0, 0.0, 1.0, 0.0, 0.0]))
    after = agent.action_scores(state, "coverage")

    assert after[3] > before[3]
    assert agent.action_credit[agent.head_names.index("coverage"), 3] > 0.0


def test_action_credit_disabled_keeps_credit_zero():
    cfg = MultiPreferenceDQNConfig(
        state_dim=5,
        action_dim=7,
        reward_dim=5,
        hidden_dim=16,
        action_credit_enabled=False,
    )
    agent = MultiPreferenceDoubleDQNAgent(cfg, device="cpu")

    agent.update_action_credit("coverage", 3, np.ones(5))

    assert np.all(agent.action_credit == 0.0)

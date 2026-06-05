import numpy as np

from rlmmo.core.niching import build_dbscan_niches
from rlmmo.optim.de import decode_action
from rlmmo.optim.individual_de import generate_individual_offspring


def test_individual_de_generates_one_trial_per_individual():
    rng = np.random.default_rng(11)
    pop = rng.random((30, 3))
    fitness = rng.random(30)
    lb = np.zeros(3)
    ub = np.ones(3)
    niches = build_dbscan_niches(pop, fitness, lb, ub)

    for action_id in range(27):
        actions = [decode_action(action_id) for _ in range(pop.shape[0])]
        offspring, owner = generate_individual_offspring(pop, fitness, niches, actions, lb, ub, rng)
        assert offspring.shape == pop.shape
        assert owner.shape == (pop.shape[0],)
        assert np.array_equal(owner, np.arange(pop.shape[0]))
        assert np.all(np.isfinite(offspring))
        assert np.all(offspring >= lb)
        assert np.all(offspring <= ub)

def test_individual_de_uses_knn_even_when_dbscan_single_cluster():
    rng = np.random.default_rng(13)
    # 点沿一维曲线密集排列，DBSCAN 很容易合成一个 cluster；测试目标是保证
    # generate_individual_offspring 不依赖 DBSCAN cluster 作为唯一候选池。
    x = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    pop = np.hstack([x, x])
    fitness = np.linspace(0.0, 1.0, 20)
    lb = np.zeros(2)
    ub = np.ones(2)
    niches = build_dbscan_niches(pop, fitness, lb, ub)
    actions = [decode_action(0) for _ in range(pop.shape[0])]
    offspring, owner = generate_individual_offspring(pop, fitness, niches, actions, lb, ub, rng, knn_k=4)
    assert offspring.shape == pop.shape
    assert np.array_equal(owner, np.arange(pop.shape[0]))
    assert np.all(np.isfinite(offspring))
    assert np.all(offspring >= lb)
    assert np.all(offspring <= ub)

import numpy as np

from rlmmo.core.niching import build_dbscan_niches
from rlmmo.optim.de import decode_action, generate_offspring


def test_de_offspring_stays_in_bounds():
    rng = np.random.default_rng(2)
    pop = rng.random((20, 3))
    fit = rng.random(20)
    niches = build_dbscan_niches(pop, fit, np.zeros(3), np.ones(3))
    actions = [decode_action(0) for _ in niches.members]
    offspring, owner = generate_offspring(pop, fit, niches, actions, np.zeros(3), np.ones(3), rng)
    assert offspring.shape == pop.shape
    assert owner.shape == (20,)
    assert np.all(offspring >= 0.0)
    assert np.all(offspring <= 1.0)

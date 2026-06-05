import numpy as np

from rlmmo.core.niching import build_dbscan_niches


def test_dbscan_assigns_every_point():
    rng = np.random.default_rng(1)
    pop = np.vstack([
        rng.normal([0.1, 0.1], 0.01, size=(10, 2)),
        rng.normal([0.9, 0.9], 0.01, size=(10, 2)),
    ])
    fit = rng.random(20)
    res = build_dbscan_niches(pop, fit, np.zeros(2), np.ones(2))
    assigned = np.concatenate(res.members)
    assert sorted(assigned.tolist()) == list(range(20))
    assert res.assigned.shape == (20,)
    assert len(res.seeds) == len(res.members)

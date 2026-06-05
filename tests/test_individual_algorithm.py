import numpy as np

from rlmmo.algorithms.online_individual_dqn_de_cmaes import (
    STATE_DIM,
    extract_individual_states,
    run_optimizer,
)
from rlmmo.core.niching import build_dbscan_niches


def test_individual_state_shape_and_finite_values():
    rng = np.random.default_rng(12)
    pop = rng.random((25, 2))
    fitness = rng.random(25)
    lb = np.zeros(2)
    ub = np.ones(2)
    niches = build_dbscan_niches(pop, fitness, lb, ub)
    states = extract_individual_states(
        pop,
        fitness,
        niches,
        lb,
        ub,
        individual_stag=np.zeros(25),
        global_stag=0,
        progress=0.25,
    )
    assert states.shape == (25, STATE_DIM)
    assert np.all(np.isfinite(states))
    assert np.all(states >= 0.0)
    assert np.all(states <= 1.0)


def test_individual_optimizer_smoke_f1_small_budget():
    result = run_optimizer(func_num=1, seed=123, np_size=20, max_fes=1000, init_method="random")
    assert result["algorithm"] == "online_individual_dqn_de_cmaes"
    assert result["FES"] <= 1000
    assert result["final_pop"].shape[1] == 1
    hist = [int(x) for x in result["dqn_action_hist"].split(";")]
    assert sum(hist) > 0

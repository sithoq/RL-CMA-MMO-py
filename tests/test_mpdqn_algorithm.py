from rlmmo.algorithms.online_individual_mpdqn_de_cmaes import run_optimizer


def test_mpdqn_optimizer_smoke_f1_small_budget():
    result = run_optimizer(func_num=1, seed=321, np_size=20, max_fes=1000, init_method="random")

    assert result["algorithm"] == "online_individual_mpdqn_de_cmaes"
    assert result["FES"] <= 1000
    assert result["final_pop"].shape[1] == 1
    assert "mp_head_hist" in result
    assert "reward_vector_mean" in result
    assert sum(int(x) for x in result["mp_head_hist"].split(";")) > 0
    assert len(result["reward_vector_mean"].split(";")) == 5

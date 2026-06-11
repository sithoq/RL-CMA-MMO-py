import pandas as pd

from rlmmo.runners.run_f1_f20 import summarize_results


def test_run_f1_f20_summary_includes_mechanism_columns(tmp_path):
    rows = [
        {
            "algorithm": "alg",
            "func_num": 6,
            "func_name": "F06",
            "family": "family",
            "dimension": 2,
            "expected_peaks": 18,
            "NP": 100,
            "maxFES": 200000,
            "run_id": 1,
            "PR": 0.50,
            "SR": 0.0,
            "FES": 200000,
            "runtime": 1.0,
            "archive_size": 100,
            "phase_switch_ratio": 0.90,
            "phase1_PR_pop": 0.20,
            "phase1_PR_archive": 0.40,
            "phase1_PR_pop_archive": 0.40,
            "phase2_PR_gain": 0.10,
            "phase2_seed_count": 12,
            "coverage_proxy": 0.50,
            "effective_archive_clusters": 9,
        },
        {
            "algorithm": "alg",
            "func_num": 6,
            "func_name": "F06",
            "family": "family",
            "dimension": 2,
            "expected_peaks": 18,
            "NP": 100,
            "maxFES": 200000,
            "run_id": 2,
            "PR": 0.70,
            "SR": 0.0,
            "FES": 200000,
            "runtime": 2.0,
            "archive_size": 100,
            "phase_switch_ratio": 0.90,
            "phase1_PR_pop": 0.30,
            "phase1_PR_archive": 0.50,
            "phase1_PR_pop_archive": 0.50,
            "phase2_PR_gain": 0.20,
            "phase2_seed_count": 14,
            "coverage_proxy": 0.60,
            "effective_archive_clusters": 10,
        },
    ]
    path = tmp_path / "F06_2runs_seed0_alg_random_test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    summary_csv, summary_md = summarize_results([path], tmp_path)
    summary = pd.read_csv(summary_csv)

    assert summary_md.exists()
    assert float(summary.loc[0, "mean_PR"]) == 0.60
    assert float(summary.loc[0, "mean_phase1_PR_pop"]) == 0.25
    assert float(summary.loc[0, "mean_phase1_PR_archive"]) == 0.45
    assert float(summary.loc[0, "mean_population_archive_gap"]) == 0.20
    assert float(summary.loc[0, "mean_phase2_PR_gain"]) == 0.15
    assert float(summary.loc[0, "mean_phase2_seed_count"]) == 13.0


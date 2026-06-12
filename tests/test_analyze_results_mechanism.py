import pandas as pd

from scripts.analyze_results import (
    build_acceptance_summary,
    collect_mechanism_rows,
    compare_mechanism_functions,
    summarize_mechanism_by_function,
)


def test_mechanism_analysis_uses_summary_totals_without_diagnostics(tmp_path):
    rows = [
        {
            "algorithm": "alg",
            "func_num": 7,
            "func_name": "F07",
            "family": "family",
            "dimension": 2,
            "expected_peaks": 36,
            "NP": 300,
            "maxFES": 200000,
            "run_id": 1,
            "PR": 0.50,
            "SR": 0.0,
            "FES": 200000,
            "runtime": 1.0,
            "archive_size": 300,
            "phase_switch_ratio": 0.90,
            "phase1_PR_pop": 0.30,
            "phase1_PR_archive": 0.50,
            "phase1_PR_pop_archive": 0.50,
            "phase2_PR_gain": 0.0,
            "phase2_seed_count": 12,
            "coverage_proxy": 0.50,
            "effective_archive_clusters": 18,
            "injection_total_count": 4,
            "injection_total_archive_gain": 1.25,
            "injection_total_to_archive_count": 2,
            "archive_reseed_total_count": 6,
            "archive_reseed_mean_fitness": 0.75,
        }
    ]
    csv_path = tmp_path / "F07_1runs_seed0_alg_random_test.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    run_df = collect_mechanism_rows(tmp_path)
    func_df = summarize_mechanism_by_function(run_df)

    assert float(run_df.loc[0, "injection_total"]) == 4.0
    assert float(run_df.loc[0, "injection_to_archive_total"]) == 2.0
    assert float(run_df.loc[0, "injection_archive_gain_total"]) == 1.25
    assert float(run_df.loc[0, "archive_reseed_total"]) == 6.0
    assert float(run_df.loc[0, "archive_reseed_mean_fitness"]) == 0.75
    assert float(func_df.loc[0, "archive_reseed_total"]) == 6.0
    assert func_df.loc[0, "failure_mode"] == "population_coverage_drift"
    assert func_df.loc[0, "acceptance_status"] == "fail"
    assert "target PR" in func_df.loc[0, "acceptance_note"]


def test_acceptance_summary_scores_core_and_hard_gates():
    baseline = pd.DataFrame(
        [
            {"func_num": 6, "func_name": "F06", "group": "core", "runs": 3, "PR": 0.65},
            {"func_num": 14, "func_name": "F14", "group": "hard", "runs": 3, "PR": 0.667},
            {"func_num": 15, "func_name": "F15", "group": "hard", "runs": 3, "PR": 0.40},
        ]
    )
    current = pd.DataFrame(
        [
            {"func_num": 6, "func_name": "F06", "group": "core", "runs": 3, "PR": 0.82},
            {"func_num": 14, "func_name": "F14", "group": "hard", "runs": 3, "PR": 0.75},
            {"func_num": 15, "func_name": "F15", "group": "hard", "runs": 3, "PR": 0.46},
        ]
    )

    compare = compare_mechanism_functions(baseline, current)
    summary = build_acceptance_summary(compare)

    assert compare.loc[compare["func_num"] == 6, "acceptance_status"].iloc[0] == "pass"
    assert compare.loc[compare["func_num"] == 14, "acceptance_status"].iloc[0] == "breakthrough"
    assert compare.loc[compare["func_num"] == 15, "acceptance_status"].iloc[0] == "improved"
    assert bool(summary.loc[summary["gate"] == "hard_no_regression", "passed"].iloc[0]) is True
    assert bool(summary.loc[summary["gate"] == "hard_at_least_two_improved", "passed"].iloc[0]) is True
    assert bool(summary.loc[summary["gate"] == "F14_F16_F18_breakthrough", "passed"].iloc[0]) is True

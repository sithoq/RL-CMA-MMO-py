# PR Improvement Workflow

This note records the current experiment loop for improving PR on CEC2013 MMO.
It is meant for server runs and post-run diagnosis, not as a final paper claim.

## Current Algorithm Branch

Use branch:

```bash
git checkout codex/coverage-rl-dbscan-improvements
git pull
```

Main algorithm:

```bash
online_individual_mpdqn_v2_de_cmaes
```

The current v2 line uses:

- conservative coverage injection
- archive quality gate for injected points
- archive-guided population reseed
- archive-prioritized multi-seed CMA-ES
- mechanism diagnostics and acceptance reports

## Function Groups

Primary acceptance does not use F8/F9 as the main tuning target.

```text
core:      F1-F7, F10-F13
hard:      F14-F20
high_peak: F8-F9, record only
focus:     core + hard
```

## Server Commands

Core functions:

```bash
python scripts/run_f1_f20.py \
  --algorithm online_individual_mpdqn_v2_de_cmaes \
  --func-group core \
  --runs 5 \
  --diagnostics \
  --out results/core_reseed_diagnostics \
  --analyze-mechanism \
  --baseline-dir results/f1_f20_diagnostics
```

Hard functions:

```bash
python scripts/run_f1_f20.py \
  --algorithm online_individual_mpdqn_v2_de_cmaes \
  --func-group hard \
  --runs 5 \
  --diagnostics \
  --out results/hard_reseed_diagnostics \
  --analyze-mechanism \
  --baseline-dir results/f1_f20_diagnostics
```

High-peak stress tests:

```bash
python scripts/run_f1_f20.py \
  --algorithm online_individual_mpdqn_v2_de_cmaes \
  --func-group high_peak \
  --runs 3 \
  --diagnostics \
  --out results/high_peak_record_only \
  --analyze-mechanism \
  --baseline-dir results/f1_f20_diagnostics
```

If diagnostics files are not available, mechanism analysis can still use the
main per-run CSV fields:

```bash
python scripts/analyze_results.py results/core_reseed_diagnostics \
  --mechanism-only \
  --baseline-dir results/f1_f20_diagnostics
```

## Acceptance Gates

Core targets:

```text
F1-F5, F10: PR >= 1.00
F6:         PR >= 0.80
F7:         PR >= 0.85
F11:        PR >= 0.95
F12:        PR >= 0.75
F13:        PR >= 0.85
```

Hard-function targets:

```text
F14-F20: no clear regression, delta_PR >= -0.02
At least two hard functions: delta_PR >= 0.05
F14/F16/F18: at least one exceeds PR=0.667
```

F8/F9:

```text
Record only. Do not tune the main algorithm around F8/F9 at this stage.
```

## First Columns To Inspect

In `server_summary_*.csv`:

```text
mean_PR
mean_phase1_PR_pop
mean_phase1_PR_archive
mean_phase1_PR_pop_archive
mean_population_archive_gap
mean_phase2_PR_gain
mean_phase2_seed_count
mean_injection_total_count
mean_archive_reseed_total_count
```

In `*_mechanism_compare_report.md`:

```text
Acceptance Summary
failure_mode_current
comparison_status
acceptance_status
acceptance_note
```

## Decision Rules

If `mean_population_archive_gap` decreases and PR increases:

```text
archive reseed is helping. Keep it and tune frequency/quality threshold only if needed.
```

If `archive_reseed_total_count > 0` but `mean_population_archive_gap` remains high:

```text
strengthen archive reseed before adding new exploration.
Possible knobs: archive_reseed_interval, archive_reseed_frac, archive_reseed_min_distance_factor.
```

If `phase1_PR_pop_archive` stays low while population/archive gap is small:

```text
this is missing-basin, not coverage drift. Add hard-function basin recovery next.
```

If `phase2_PR_gain` is high:

```text
preserve CMA-ES budget and archive-prioritized seed selection.
```

If `mean_injection_total_count` is high but PR does not improve:

```text
injection is noisy. Keep conservative mode and tighten archive quality gate.
```

If core functions regress:

```text
stop tuning hard functions and restore core stability first.
```


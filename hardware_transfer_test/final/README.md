# Final — the active deliverable

Everything you need for the chosen genome family, `plain_seed123`, in one place. For
everything else that was tried and set aside, see `../archive/` (and its own README) and
`BEAT_BASELINE_INVESTIGATION_LOG.md` below for why it was set aside.

## Genomes and their generated artifacts

- **`plain_seed123/`** — unclamped, best simulation performance. `genome_trained_n20.npy` +
  per-`n_agents` (1,2,3,4,5,7,10,20) `video.mp4`/`battery_plot.png`/`metrics.json`.
- **`plain_seed123_clamped/`** — retrained from scratch with the hard safety clamp, the one
  actually deployed to hardware (`ants26_replication/hardware_deployment/`). Same artifact
  layout.
- **`lj_baseline/`** — the rule-based comparison baseline, same artifact layout.

## The story and the numbers

- **`BEAT_BASELINE_INVESTIGATION_LOG.md`** — start here. Full history: the search, the
  bimodal seed pattern, the n_agents-sweep finding, the wall-sticking discovery that decided
  the final choice, and the safety-clamp investigation.
- **`overleaf_summary/summary_plain_seed123.tex`** — the methodology + results write-up
  (both variants), ready for Overleaf. Figures in `overleaf_summary/figures/`.
- **`leadership_analysis/`** — turn-taking metrics (reciprocity, hierarchy, front-occupancy,
  persistence-filtered hand-offs) for the three conditions above, plus `summary_table.json`
  and `seed_sweep_table_n{10,20}.json` for the raw per-seed numbers.

## Scripts (all runnable from within this directory: `cd final/ && python <script>.py`)

| Script | Produces |
|---|---|
| `leadership_metrics.py` (+ `leadership_plots.py`) | Everything in `leadership_analysis/` |
| `paper_figures.py` | Fig.~5a/6 analogs in `overleaf_summary/figures/` |
| `n_agents_sweep_analysis.py` | Distance/battery/position-change/battery-equity vs. swarm size |
| `generate_plain_seed123_clamped.py` | Re-generates `plain_seed123_clamped/`'s videos/metrics |
| `safety_clamp_figure.py` | The clamp scale-function figure used in the write-up |

**Note**: running any of these overwrites its own output with fresh data — if you only need
to inspect existing numbers, read the JSON/figures directly rather than re-running.

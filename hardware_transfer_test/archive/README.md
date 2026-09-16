# Archive

Everything in this directory is a genome, ablation, or analysis script from earlier in the
"beat the LJ baseline" investigation that is **not** part of the current deliverable — kept
for reference and comparison, not because it's still being used. For the active,
currently-relevant material, see `../final/` instead. For the full story of how the
investigation got from here to there, see `../final/BEAT_BASELINE_INVESTIGATION_LOG.md`.

## What's here

**Original paper-replication genomes** (pre-dates the "beat baseline" search):
`pre_clamp_best/`, `safety_clamp_best/`.

**Curriculum-ablation genomes** (`walk_left` vs. `walk_upwind`, clamp on/off, all `n=10`):
`upwind_clamp/`, `upwind_no_clamp/`, `walk_left_n10_clamp/`, `walk_left_n10_no_clamp/`.

**The two candidate genomes from the winning 2-stage recipe that were *not* chosen** in the
end (both had a higher both-beat-baseline rate than `plain_seed123` in simulation, but both
chronically hug/stick to a wall at hardware-relevant swarm sizes — see the investigation log
§4/§5 for why that ruled them out): `upwind_2stage_plain_seed888/` (+ `_clamped`),
`upwind_2stage_drain0_seed123/`.

**`lj_baseline/`** — a full copy of the rule-based comparison baseline, duplicated here (also
present in `../final/`) so this archive stays self-contained and every script below can still
find its sibling dependency without reaching into `../final/`.

**Scripts**, each analyzing/generating artifacts for the genomes above (all still runnable —
their `REPO_ROOT`/path logic was fixed for this directory's depth during the 2026-09-16
reorganization, but they have not been re-verified beyond an import-level check):
`leadership_metrics.py` + `leadership_plots.py` → `leadership_analysis/` (the original
pre_clamp/safety_clamp/lj_baseline turn-taking analysis), `paper_figures.py`,
`compare_all_conditions.py`, `compare_zero_drain_universal.py`, `n_agents_sweep_analysis.py`
(the version covering all three original 2-stage-upwind candidates, not just
`plain_seed123`), `generate_new_conditions.py`, `generate_upwind_2stage_conditions.py`,
`generate_plain_seed888_clamped.py`, `beat_baseline_final_comparison.py`.

`summary.json` here is the old, shared aggregation file written by several of the scripts
above across the whole session — a historical snapshot, not kept in sync with anything in
`../final/` anymore.

## If you need to resurrect something from here

The genome `.npy` files and any already-generated videos/plots need nothing — just read them.
If you want to *re-run* one of the scripts, it should work as-is from within this directory
(`cd archive/ && python <script>.py`); if something's still off, the most likely cause is a
path assumption from before the reorg — compare against the equivalent script in `../final/`
for the pattern that's known to work.

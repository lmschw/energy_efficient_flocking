# "Beat the LJ baseline" investigation — full log and final decision

This document is the durable record of a multi-day investigation that started from a single
observation: the originally-evolved Hebbian controllers (`pre_clamp_best`, `safety_clamp_best`)
covered the same distance as the rule-based LJ baseline while burning **more** battery — i.e.
no reason to prefer the evolved controller at all. Everything below traces how that got fixed,
what didn't work, and why the codebase ended up where it did. Written so either of us can pick
this up cold later without re-deriving it.

**Final decision (2026-09-16): `plain_seed123`**, in two forms:
- **Unclamped** (`upwind_2stage_plain_seed123/genome_trained_n20.npy`) — best simulation-only
  numbers, for the paper's simulation-results claims.
- **Clamped, retrained from scratch** (`upwind_2stage_plain_seed123_clamped/genome_trained_n20.npy`)
  — collision-safe, for the real hardware trial.

Chosen over the other 3 candidates that emerged from the same search specifically because it's
the only one that doesn't chronically hug/stick to a wall (see "The wall-sticking discovery"
below) — a deciding factor that only surfaced after visually reviewing the generated videos,
not something the distance/battery/both-beat numbers alone would have caught.

## 1. The search for a baseline-beating recipe

Roughly thirty training/evaluation configurations were tried before finding one that worked
reliably (not just on average). In rough chronological order:

| # | Idea | Outcome |
|---|---|---|
| 1-4 | Zero/reduced collision penalty, collision-cost-only (no battery drain) variants | Volatile means, 10-20% both-beat-baseline reliability at best |
| 5-10 | "Drain holiday" while colliding, swept 0/10/25/50/75% refund | Same volatile pattern |
| 11-12 | Doubled wind-drag coupling (κ), alone and with drain-holiday | Widens exposure inequality between agents (real effect) but insufficient alone |
| 13 | Reverted to 2-stage curriculum (`walk_upwind` → `save_battery_avoid_all` directly, skipping the wall-only stage) at n=10 | First configuration to look meaningfully different |
| **14-28** | **2-stage-upwind at n=20**, 5 seeds (42/123/777/2026/888) × 2 collision-cost variants (plain drain, 0%-drain-holiday) | **First reliable wins** — see below |

Two bugs were caught and fixed *during* this search, before they corrupted any reported
result: (1) a stall in an early drain-holiday variant where near-zero net drain let episodes
run ~20h before a `MAX_STEPS=5000` cap was added; (2) a follow-up exploit where the same
refund mechanic let a permanently-colliding cluster pay effectively zero net drain — fixed by
only refunding drain *above* the idle-power floor, giving a hard mathematical ~1000-step bound.

### The bimodal seed pattern

Both the plain-drain and 0%-drain-holiday recipes turned out to be **bimodal across training
seeds**, not uniformly good:
- Plain drain (5 seeds): only 123 and 888 reached the ~83% both-beat tier; 42/777/2026 landed
  at 10-23% (no better than every earlier attempt).
- 0%-drain-holiday (5 seeds): only 123 reached a high tier (97-100% depending on eval
  protocol); 42/777/2026/888 landed at 10-63%.

**Training-seed selection mattered as much as the recipe itself.**

## 2. The three original winners (n=20, standard/real drain physics, 30 seeds vs. n=20 LJ baseline)

| genome | distance | battery | both-beat |
|---|---|---|---|
| LJ baseline (n=20) | 8.81 ± 1.02m | 18.68 ± 5.46% | — |
| `upwind_2stage_plain_seed123` | 19.46 ± 2.25m | 23.06 ± 5.02% | 25/30 (83%) |
| `upwind_2stage_plain_seed888` | 21.06 ± 5.52m | 28.74 ± 11.31% | 25/30 (83%) |
| `upwind_2stage_drain0_seed123` | 13.44 ± 0.88m | 41.48 ± 6.12% | 29/30 (97%) |

Note the n=20 LJ baseline is a much lower bar than the n=10 one (21.00m/24.57%) — the
baseline's own distance collapses by more than half at the larger swarm size. This is disclosed
explicitly rather than left implicit in any "beats baseline" claim.

## 3. The n_agents-sweep finding: the win does NOT generalize below n=20

A 30-seed sweep down to n_agents=1..10 found **zero of 30 seeds** where any of the three
winners beat the LJ baseline on both axes at any swarm size below 20 — the reliable win only
appears at n=20. The LJ baseline's distance is roughly flat (~25m) through n=7 and only
collapses toward n=10-20, while the genomes climb steadily and only cross over near the top of
the range. **This directly matters for hardware deployment**: a real trial with only a handful
of robots would not be expected to reproduce the "beats baseline" result at all.
(`overleaf_summary/n_agents_sweep_comparison.json`, `figures/n_agents_sweep_distance_battery.*`.)

## 4. The wall-sticking discovery

Not captured by any of the metrics above — found by watching the generated videos and then
confirmed against an already-computed but previously-unsurfaced field
(`wall_collision_time`/`wct` in every `metrics.json`). As a percentage of total agent-time
budget (n_agents × episode duration) spent within contact margin of a wall:

| n_agents | plain_seed123 | plain_seed888 | drain0_seed123 |
|---|---|---|---|
| 1 | 0.0% | 0.0% | 47.1% |
| 2 | 0.0% | 0.8% | 70.1% |
| 5 | 6.7% | 12.0% | 67.9% |
| 10 | 0.0% | 48.6% | 71.6% |
| 20 | 0.0% | 44.6% | 58.0% |

`plain_seed123` is essentially clean. `drain0_seed123` sticks to walls even completely alone
(n=1) — not a multi-agent artifact, it just likes walls. `plain_seed888` is fine at low n but
degrades badly as swarm size grows. None of the genomes have any wall-distance sensory input
at all (confirmed in `ants26_replication/hardware_deployment/README.md`) — this behavior is
pure incidental reward-shaping correlation from training in a fixed-size arena, not a real
spatial sense, so it can differ wildly and unpredictably per seed. **This was the deciding
factor for choosing `plain_seed123` over `drain0_seed123` despite the latter's higher
both-beat rate (97% vs 83%)** — a genome parked at a wall for 50-70% of its runtime is a bad
real-hardware candidate regardless of its simulated distance/battery numbers.

## 5. The safety-clamp investigation

None of the three winners were trained with the hard forward-speed collision-avoidance clamp
(`HEBBIAN_SAFETY_CLAMP_ENABLED` etc.) active — a real concern for physical robots with no
compliance.

**Bolt-on governor: tested and rejected.** A deployment-only speed governor (mirroring the
existing corridor-wall governor's "scale v, never touch w" pattern) was tested at six band
widths from the training-grade 0.30/0.05m gap down to a razor-thin 0.005/0.001m near-contact-
only band. **Every width collapsed distance to ~10-16% of unclamped** via the same gridlock
failure mode already documented for the original pre-clamp genome — any discrepancy between
commanded and executed speed appears to derail the online Hebbian update into a bad fixed
point, regardless of how rarely it's triggered. No band width is viable. **There is no cheap
deployment-side fix for collision safety on these genomes.**

**Retrained from scratch instead** (`ants26_replication/upwind_safety_variant/run_2stage_upwind_clamped.py`,
`run_2stage_upwind_drain_holiday_clamped.py`), matching how `safety_clamp_best` was originally
produced — all 3 winning seeds, ~11-13h wall-clock each. Stage-1 (distance-only) cost was
uneven: `plain_seed123` barely affected (211→203 raw efficiency), `plain_seed888`/`drain0_123`
lost ~60% (413→148, 475→193) — the clamp is far more expensive for genomes whose unclamped
strategy depended on tight/fast clustering.

**Final clamped 30-seed eval** (clamp active, standard drain physics, vs n=20 LJ baseline):

| genome | dist: unclamped → clamped | batt: unclamped → clamped | both-beat: unclamped → clamped |
|---|---|---|---|
| plain_seed123 | 19.46m → 14.06m | 23.06% → 15.77% | 83% → **17%** |
| plain_seed888 | 21.06m → 9.30m | 28.74% → 29.97% | 83% → **73%** |
| drain0_seed123 | 13.44m → 13.40m | 41.48% → 16.48% | 97% → **30%** |

None preserve their unclamped reliability, each via a different failure mode. Taken alone,
`plain_seed888` clamped looked like the best collision-safe survivor (73% both-beat) — **but
it's also the one with the worst wall-sticking problem at exactly the swarm sizes a real trial
would use.** Combining both findings is why `plain_seed123` (clamped, 17% both-beat but clean
wall behavior) was chosen for hardware over `plain_seed888` (clamped, 73% both-beat but bad
wall behavior) — collision safety and "doesn't get stuck at a wall" both being closer to hard
requirements for a physical trial than "beats baseline," which the n_agents-sweep finding
already showed wouldn't manifest at real-trial swarm sizes anyway.

One more data point in `plain_seed123`'s favor: the clamp does not suppress its position-
changing (front-rank hand-off) behavior — it actually increases it (raw switches 24→55, real/
persistence-filtered switches 10→23 at n=20, seed 42), and its clean wall behavior (13s of
wall-contact time at n=20, vs. hundreds-to-low-thousands for the others) survives the clamp too.

## 6. Where everything lives

**Reorganized 2026-09-16** into `hardware_transfer_test/final/` (everything below — the
active deliverables) and `hardware_transfer_test/archive/` (every other candidate/ablation
genome and the scripts that only concern them, kept for reference but not part of the current
story). If you're looking for `pre_clamp_best`, `safety_clamp_best`, `upwind_clamp`,
`walk_left_n10_*`, `plain_seed888`, or `drain0_seed123` — they're all in `archive/` now, each
with its own copy of `lj_baseline/` and (for the pre-2-stage-upwind era ones) the original
`leadership_analysis/` pipeline, so `archive/` is self-contained and doesn't depend on
anything in `final/`.

**Genomes** (all `n_agents=20`-trained), in `hardware_transfer_test/final/`:
- `plain_seed123/genome_trained_n20.npy` — unclamped, chosen for simulation results
- `plain_seed123_clamped/genome_trained_n20.npy` — clamped, chosen for hardware
- `lj_baseline/` — the rule-based comparison baseline (also duplicated into `archive/` so
  that side stays self-contained too)
- Source training runs (unchanged location): `results/hebbian_results_v2_2stage_upwind{,_drain_0pct}{,_clamped}/n20_seed{123,888,777,2026,42}/`
  — includes the archived seeds/recipes too, not reorganized (this is the raw CMA-ES output,
  not a `hardware_transfer_test` deliverable)

**Training scripts** (unchanged location): `ants26_replication/upwind_safety_variant/run_2stage_upwind{,_clamped}.py`,
`run_2stage_upwind_drain_holiday{,_clamped}.py`.

**Isolated analysis for the final choice**, all in `hardware_transfer_test/final/` (suffix
`_plain_seed123` dropped from script names now that it's the only variant in this folder --
still present in some output filenames for consistency with what's already cited in the
write-up):
- `leadership_metrics.py` / `leadership_plots.py` → `leadership_analysis/`
- `paper_figures.py` → Fig 5a/6 analogs in `overleaf_summary/figures/*_plain_seed123.*`
- `n_agents_sweep_analysis.py` → `overleaf_summary/n_agents_sweep_comparison_plain_seed123.json` + figures
- `generate_plain_seed123_clamped.py` → per-`n_agents` videos/battery-plots/metrics for the clamped variant
  (the unclamped variant's artifacts already exist under `plain_seed123/n*/`, generated
  before the reorg by the now-archived `generate_upwind_2stage_conditions.py`)
- `safety_clamp_figure.py` → the clamp scale function figure used in the write-up

**Write-up**: `overleaf_summary/summary_plain_seed123.tex` (methodology + results, both
variants) -- standalone, does not depend on `archive/`'s `summary.tex`.

**Hardware deployment**: `ants26_replication/hardware_deployment/` (not reorganized, separate
concern) — see that directory's own README.md for the deployment-specific decision and setup
steps.

**Memory** (Claude's own persistent notes, for continuity across sessions):
`project_upwind_safety_variant.md` — the working log this document was distilled from, with
finer-grained per-job detail if something here needs double-checking.

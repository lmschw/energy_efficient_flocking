# energy_efficient_flocking

- `ants26_replication/` -- current work: the Hebbian ABCD replication of Mahdavi et al.'s
  ANTS 2026 paper, plus hardware deployment for real Thymio+Pi swarms.
- `initial_implementation/` -- the original LJ-force flocking model this project grew
  out of. Superseded by ants26_replication/ but kept for reference/comparison.
- `crazythymio/` -- hardware stack for the next round of hardware experiments: Thymio +
  Crazyflie board with Lighthouse positioning in place of OptiTrack. Upstream code from
  https://github.com/fudavd/CrazyThymio (see [Acknowledgements](#acknowledgements)); our
  self-contained deployment of the Hebbian controller on it (own config, custom firmware,
  launcher, setup/calibration guide) is in `crazythymio/lighthouse_deployment/` and
  `crazythymio/firmware/` -- start with `crazythymio/lighthouse_deployment/README.md`.
  Independent of the OptiTrack deployment in `ants26_replication/hardware_deployment/`.
- `hardware_transfer_test/` -- evaluates three conditions side by side (the LJ Table-3
  rule-based baseline, and two evolved Hebbian genomes -- one from before the hard
  safety-clamp reflex layer existed, one trained with it active) across n_agents in
  {2,3,4,5,10,20}: archived per-run `metrics.json`/`video.mp4`, plus
  `leadership_metrics.py`/`leadership_plots.py`/`paper_figures.py` (turn-taking and
  leadership-network analysis, reproducing each condition's trajectory from its saved
  genome/rules -- see that directory's own docstrings for the exact reproduction recipe)
  and `overleaf_summary/` (a ready-to-compile LaTeX methodology + results write-up with
  figures as PDF/SVG/PNG).

`ants26_replication/` and `initial_implementation/` are standalone Python projects (each
its own `experiment/` package) that share no imports, only some physics constants happen
to have the same values by necessity (see `ants26_replication/experiment/config.py`'s
docstring). `hardware_transfer_test/`'s analysis scripts are the one exception: they
import directly from `ants26_replication/experiment/` rather than duplicating it.

## Acknowledgements

`crazythymio/` contains code from [fudavd/CrazyThymio](https://github.com/fudavd/CrazyThymio)
(Fuda van Diggelen et al., MIT License, copied unmodified along with its `LICENSE`), the
codebase of *Learning online adaptive emergent control for heterogeneous robot swarms*. It
is used here for the Thymio/Raspberry Pi/Crazyflie-board interface (state estimate from the
Lighthouse positioning system, IR-based quadrant sensing, motor commands). The upstream
simulation submodule (`sim_exp`), firmware submodule (`CrazyThymio-firmware`, from
https://github.com/tugayalperen/CrazyThymio-firmware) and the `results/` data were not
copied; see the upstream repository for those. `crazythymio/firmware/app_share_pos_hebbian/`
is derived from the firmware's `app_share_pos` example (GPL-3.0).

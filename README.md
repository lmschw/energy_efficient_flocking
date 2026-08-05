# energy_efficient_flocking

- `ants26_replication/` -- current work: the Hebbian ABCD replication of Mahdavi et al.'s
  ANTS 2026 paper, plus hardware deployment for real Thymio+Pi swarms.
- `initial_implementation/` -- the original LJ-force flocking model this project grew
  out of. Superseded by ants26_replication/ but kept for reference/comparison.

Each directory is a standalone Python project (its own `experiment/` package); they
share no imports, only some physics constants happen to have the same values by
necessity (see `ants26_replication/experiment/config.py`'s docstring).
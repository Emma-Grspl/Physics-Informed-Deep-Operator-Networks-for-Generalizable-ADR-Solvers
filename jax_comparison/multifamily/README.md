# Multifamily Comparison

This subtree contains the strict PyTorch vs JAX comparison on the full ADR task with all initial-condition families.

This is the main comparison result of the repository.

## Purpose

The multifamily track asks:

- if both frameworks are given the same overall training protocol, which one produces the usable ADR surrogate on the real three-family problem?

## Contents

- `configs/`: PyTorch, JAX, and benchmark configs for the strict comparison
- `launch/`: SLURM launchers for the matched multifamily runs
- benchmark helpers and scripts referenced by those launchers

## Interpretation

This directory should be treated as the primary comparison evidence.

Compared with `monofamily/`, the conclusions here carry more weight because they evaluate the actual target problem rather than a simplified diagnostic setting.

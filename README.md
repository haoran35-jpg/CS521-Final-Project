# CS521-Final-Project: Dual-Interval Analysis for JAX jaxpr

Static Analysis of Automatic Differentiation via Dual-Intervals on JAX jaxpr

## Overview

This project implements dual-interval arithmetic for statically analyzing programs represented by the jaxpr (JAX expression) intermediate representation. The analysis provides sound overapproximations of both value and gradient bounds.

## Visualization

To recreate our visualization graphs, use the command line:

```bash
python visualize_bounds.py
python visualize_gpt2_bounds.py
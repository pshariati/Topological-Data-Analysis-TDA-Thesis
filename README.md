# Topological Data Analysis (TDA) Thesis

This repository contains the code and final thesis for my **Tufts University Senior Honors Thesis (May 2021)**, completed under the supervision of the Department of Mathematics.

📄 **Thesis PDF:** `A_New_Comparison_Metric_for_Computational_Topology.pdf`

## Overview
This project introduces a novel comparison framework for Mapper outputs in **computational topology** using **metric measure space theory** and the **Gromov–Wasserstein (GW) metric**. The approach provides a quantitative and robust method for comparing shapes and detecting topological divergence in noisy datasets.

## Contents
- `thesis_code.ipynb` — Core notebook for data generation, Mapper construction, and Gromov–Wasserstein distance computation.  
- `A_New_Comparison_Metric_for_Computational_Topology.pdf` — Full thesis describing the theory, implementation, and experimental results.  

## Key Results
- Demonstrated **robustness to Gaussian noise** up to σ ≤ 0.315 on synthetic data (e.g., noisy circles).  
- Applied the framework to **Chicago 2015 mayoral election precinct voting data**, detecting topological divergence at σ = 0.08.  
- Introduced a **metric-based visualization technique** (GW heatmaps) to compare Mapper outputs.

## Citation
If referencing this work, please cite:

> Shariati, Pejmon (2021). *A New Comparison Metric for Computational Topology.* Senior Honors Thesis, Tufts University.

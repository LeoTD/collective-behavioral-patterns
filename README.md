# Collective Behavioral Patterns at Different Scales

This repository contains the primary dataset and additional testing reports from the publication:

**Collective Behavioral Patterns at Different Scales**  
Marcela Camacho, Mateo Zuleta, Leonardo Tanenbaum-Diaz, Dario Dominguez, Mariela Marin, Davy Risso, Andrew Macvean, Patricia Diaz.

## Overview
This repository houses the electrophysiological time series data (macrophage macroscopic ion currents) demonstrating fractal behavior and memory, as detailed in the *Collective Behavioral Patterns at Different Scales* paper. Additionally, it contains the generated algorithms from **AlphaEvolve**—an evolutionary coding agent—including the highest-performing ensemble estimator, the **Hyper-Dimensional Fractal Consensus (HDFC)**. HDFC was evaluated across synthetic processes (fBm, fGn, Levy flights), real-world macrophage recordings, and financial market data (S&P 500).

## Requirements
Python 3.12+ is recommended. To install the required dependencies for running the HDFC estimators, please run:
```bash
pip install -r requirements.txt
```

## Directories
*   **`datasets_macrophages/`**: Primary experimental data. Contains electrophysiological recordings (.dat and .json formats) of control murine macrophage-like cell line J774.A1 in the whole-cell configuration.
*   **`alpha_evolve_generation/`**: Contains the scripts used to interface with the AlphaEvolve LLM-driven coding agent, the evolutionary loop logic, and the scoring functions minimizing Mean Squared Error (MSE).
*   **`hdfc_estimator/`**: Contains the frozen (HDFC-E), instrumented (HDFC-I), and revised production-grade (HDFC-R) code for HDFC, which achieved a top MSE of 0.0010912 against the ground truth.
*   **`additional_testing_reports/`**: Synthesis reports and Jupyter notebooks detailing HDFC's performance on:
    *   Synthetic benchmarks (fBm, fGn, fLm).
    *   Biophysical validation (Macrophage cell trajectories).
    *   Financial applications (Regime-switching trading strategy backtests on SPY).

## Workflow
1.  **Generate / Process the Macrophage Data**:
    Run the preprocessing script to convert raw `.dat` patch-clamp recordings to `.json` format used by the evolutionary agent.
    ```bash
    python scripts/preprocess_cell_recordings.py
    ```
2.  **Run AlphaEvolve**:
    To initialize the evolutionary coding agent loop to evolve Hurst exponent estimators:
    ```bash
    python alpha_evolve_generation/run_evolution.py --generations 100 --population 500
    ```
    This process evaluates the candidate programs against the provided cell recordings exhibiting documented fractal behavior.
3.  **Evaluate HDFC on Synthetic and Financial Data**:
    Use the included scripts to run HDFC-R (Revised, $O(N)$ speed) against the synthetic benchmarks or the S&P 500 backtest to analyze its predictive value in non-stationary markets.
    ```bash
    python hdfc_estimator/evaluate_synthetic.py
    python hdfc_estimator/financial_backtest.py
    ```

## Data Sources & Background
*   **Macrophage Electrophysiology**: Recordings of macroscopic K+ currents elicited by voltage steps, demonstrating long-memory dynamics, hysteresis, and a slower-than-expected inactivation. These characteristics align with an equivalent fractional RC circuit model.
*   **Algorithmic Synthesis**: AlphaEvolve generated over 27,550 programs. The resulting HDFC algorithm combines Singular Value Decomposition (SVD) Decay, Phase-Space Surrogates, and Temporal PCA ("Wormhole") into a coherent consensus engine to robustly estimate the Hurst exponent ($H$).

## License
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
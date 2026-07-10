# Collective Behavioral Patterns at Different Scales

This repository contains the primary dataset and additional testing reports from the publication:

**Collective Behavioral Patterns at Different Scales**  
Marcela Camacho, Mateo Zuleta, Leonardo Tanenbaum-Diaz, Dario Dominguez, Mariela Marin, Davy Risso, Andrew Macvean, Patricia Diaz

## Overview
This repository houses the electrophysiological time series data (macrophage macroscopic ion currents) demonstrating fractal behavior and memory, as detailed in the *Collective Behavioral Patterns at Different Scales* paper. Additionally, it contains the generated algorithms from **AlphaEvolve**—an evolutionary coding agent—including the highest-performing ensemble estimator, the **Hyper-Dimensional Fractal Consensus (HDFC)**. HDFC was evaluated against standard estimators using synthetic processes (fBm, fGn, Levy flights), real-world macrophage recordings, and financial market data (S&P 500). Included are the reports detailing the results.

## Requirements
Python 3.12+ is recommended. To install the required dependencies for running the HDFC estimators and standard estimators, please run:
```bash
pip install -r requirements.txt
```

## Directories
*   **`macrophage_dataset/`**: Primary experimental data. Contains electrophysiological recordings (.json formats) of control murine macrophage-like cell line J774.A1 in the whole-cell configuration of the patch clamp technique.
*   **`estimators/`**: Contains the frozen (HDFC-E), instrumented (HDFC-I), and revised (HDFC-R) code for HDFC. Also contains a range of standard estimators used for testing.
*   **`additional_testing_reports/`**: Containts reports detailing HDFC's performance across a range of tests, contrasted against other standard methods.

## License
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

The original macrophage dataset was acquired by researchers at the  Biophysics and Biology of Membranes, Laboratorio de Biofísica, Departamento de Biología, Facultad de Ciencias, Universidad Nacional de Colombia 111321, and can be used according to [Acuerdo 025 2003 of Universidad Nacional de Colombia](https://legal.unal.edu.co/rlunal/home/doc.jsp?d_i=34248)

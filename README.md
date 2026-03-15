# Observational Exclusion of a Minimal k-Essence Inflationary Model

This repository provides the numerical pipeline used to establish the observational exclusion of a class of k-essence inflationary models with $\phi^{-4}$ kinetic coupling, as presented in the paper [Link to Paper/arXiv].

## 📋 Overview
We investigate the phenomenological boundaries of a k-essence model featuring:
- A $\phi^{-4}$ kinetic coupling motivated by Weyl non-metricity.
- A horizon-bounded logarithmic potential.
- An algebraic master equation $u(1+2u)^2 = \mathcal{C}F(y)$.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

### Usage
Run the main benchmark script to solve the master equation and generate the observable tables:
```bash
python src/compute_benchmarks.py
📊 Key Results
The script reproduces Figures 1-5 from the manuscript, specifically:
Master Equation Attractors: Verification of the e^{-3N} damping rate.
Observational Constraints: Mapping the predicted r \approx 0.14 against Planck/BICEP data.

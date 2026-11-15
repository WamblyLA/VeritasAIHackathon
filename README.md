# SensorGuard
Track: AI + Climate

Team: Mathanalysis

## The problem
City services depend on PM2.5 stations to track air pollution
Sensor failures create gaps that can hide pollution peaks and complicate maintenance decisions

## What we built
A system that reconstructs missing PM2.5 readings from available stations, simulates continuous sensor failures and identifies priority stations for maintenance
The application uses saved models and real held-out measurements

## What we added
Our own idea
We study missing-reading reconstruction rather than ordinary pollution forecasting, compare short and continuous outages, and optimize which stations remain operational

## How it works
UCI Beijing Multi-Site Air Quality provides hourly observations from 12 stations
Models include MDAE, Extreme-aware MDAE, official PyPOTS SAITS and an 80% SAITS + 20% Extreme-aware MDAE ensemble
Train: March 2013 to December 2015; validation: January to June 2016; test: July 2016 to February 2017, with unseen measurements artificially hidden
Normalization uses training data only; genuine gaps are never labels; temporal inputs contain 24 past hours plus the current observed hour, without future readings
Python 3.12 and an NVIDIA CUDA GPU are required
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m src.evaluation
```
The evaluation command restores missing data and models, checks leakage and continuous outages, and repeats validation-only station selection
Missing checkpoints train automatically with seed 2026; existing checkpoints load without retraining

## Results
High-pollution events are defined as PM2.5 $\geq$ 187 µg/m³, the 90th percentile of training observations; all models share 1,130,940 hidden targets across repeated outage scenarios
| Model | MAE | RMSE | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Ridge | 18.640 | 33.430 | 82.88% | 79.88% | 0.8136 |
| MDAE | 16.744 | 30.727 | 84.18% | 81.92% | 0.8303 |
| Extreme-aware MDAE | 17.840 | 31.793 | 77.62% | 88.70% | 0.8279 |
| SAITS | 13.592 | 26.714 | 89.71% | 81.99% | 0.8567 |
| MDAE + history | 15.745 | 28.099 | 86.03% | 84.72% | 0.8537 |
| Ensemble | 13.408 | 25.844 | 88.68% | 84.55% | 0.8657 |


The validation-selected ensemble improves MAE by 1.35% over SAITS; temporal models receive additional history, so this is not an architecture-only comparison
On paired outage endpoints, SAITS MAE rises from 9.90 at 1h to 18.01 at 168h; the ensemble rises from 9.94 to 16.83, while current-only MDAE stays at 14.66
These are retrospective simulations on one known benchmark, not evidence that stations can safely be removed

## What we'd do next
Validate adaptive model switching during prolonged failures and test on new cities and chronological holdouts

## Sources
[UCI Beijing Multi-Site Air Quality](https://doi.org/10.24432/C5RK5G), Chen (2017), CC BY 4.0
[SAITS](https://doi.org/10.1016/j.eswa.2023.119619) and [official PyPOTS](https://github.com/WenjieDu/PyPOTS), BSD-3-Clause
[PyTorch](https://pytorch.org), [Streamlit](https://streamlit.io) and [Plotly](https://plotly.com/python/)

# DMH-LSTM: A Real-Time Probabilistic Framework for Operational Crude Oil Price Forecasting

This repository provides the official Python implementation of the **DMH-LSTM** framework.  
The proposed system bridges the gap between deep learning research and real-world operational energy forecasting by enforcing a **strict leakage-free, real-time pipeline** and delivering **probabilistically calibrated forecasts**.

---

## Project Overview

This repository hosts the code for the **DMH-LSTM (Direct Multi-Horizon Long Short-Term Memory)** framework, a deep learning system specifically designed for **operational crude oil price forecasting**.

Unlike standard offline forecasting approaches that rely on static train–test splits and may suffer from look-ahead bias, this framework implements a **walk-forward, limited-information setting** that closely mimics real-world deployment conditions. The methodology is tailored to capture the strong non-linearities and volatility regimes typical of energy markets, providing not only accurate point forecasts but also **well-calibrated uncertainty intervals**, which are essential for risk-aware decision-making.

---

## 🚀 Key Features

- **Direct Multi-Horizon (DMH) Forecasting**  
  Independent prediction heads are trained for each forecast horizon, avoiding recursive error accumulation.

- **Operational Walk-Forward Pipeline**  
  A strict rolling backtesting scheme that fully eliminates look-ahead bias and ensures realistic performance evaluation.

- **Probabilistic Forecasting via Conformal Prediction**  
  Distribution-free conformal prediction methods are integrated to generate reliable prediction intervals, even under extreme volatility conditions.

- **Model Interpretability**  
  Gradient-based **saliency maps** are used to assess the relative importance of temporal lags in the forecasting process.

- **Financial Risk Metrics**  
  Forecasts are evaluated not only through standard accuracy measures but also using **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**.

---

## 📊 Data Description

The empirical analysis focuses on the **Mexican Mix crude oil price**.

- **Variable:** Mexican Mix (MM) crude oil price  
- **Source:** Official data from the Bank of Mexico  

Due to data licensing and commercial restrictions, the raw price series cannot be publicly redistributed.

To ensure transparency and reproducibility, the repository includes **machine-readable intermediate outputs** (e.g. backtesting results in `.npz` format) that allow full reproduction of the forecasting and uncertainty quantification pipeline without requiring access to the original raw data.

---

## 🧠 Reproducibility and Code Structure

All models and methods used in this study are **developed in-house** and provided as **lightweight, modular Python implementations**.

- All core functionality (preprocessing, model architecture, training, backtesting, and conformal prediction) is implemented in standalone `.py` files under `src/`.
- **No core functions are defined inside Jupyter notebooks.**
- The folder `jupyter_notebook/` contains **a single illustrative notebook** that:
  - Loads precomputed, machine-readable backtesting outputs.
  - Calls the corresponding functions implemented in `src/`.
  - Reproduces the main empirical results reported in the paper.

This design choice was made explicitly to:
- Avoid hidden logic inside notebooks,
- Facilitate code inspection and reuse,
- Eliminate any ambiguity regarding result generation.

No proprietary software or large-scale pretrained models are required.

---

## 🧑‍💻 Main Author

**Juan Adrian Moreno Hernández**  
Doctoral Program in Energy  
Escuela Superior de Ingeniería Mecánica y Eléctrica (ESIME)  
Instituto Politécnico Nacional (IPN), Mexico City, Mexico  

---

## 📚 Citation

If you use this repository or its results, please cite:

> Moreno-Hernández, J. A., De la Portilla-Reynoso, M., & Moreno-Hernández, R. C. (2026).  
> *A Real-Time Probabilistic Direct Multi-Horizon LSTM Framework for Operational Crude Oil Price Forecasting*.  
> Instituto Politécnico Nacional (IPN).

---

## 👥 Acknowledgments

This work was carried out with academic supervision from:

- **Dr. José Alfredo Jiménez-Bernal**, ESIME-IPN  
- **Dr. Didier Samayoa-Ochoa**, ESIME-IPN  
- **Dr. Claudia del Carmen Gutiérrez-Torres**, ESIME-IPN

---

## 📁 Data Availability Statement

The raw data used in this study (Mexican Mix crude oil prices) are subject to commercial and institutional restrictions and therefore cannot be publicly shared.

In compliance with the data availability policies of *Energy Economics*, this repository provides **machine-readable intermediate outputs** (e.g., backtesting results stored in `.npz` format) that fully preserve the information required to reproduce:

- the walk-forward training and evaluation scheme,
- the multi-horizon point forecasts,
- the conformal prediction intervals,
- and all reported accuracy, coverage, and risk metrics.

These outputs enable independent verification of the methodology and results without requiring access to the proprietary raw price series.

---

## 🔁 Computational Reproducibility Statement

All computational components of the proposed framework were developed **in-house** and are fully provided in this repository.

- All core methods are implemented in modular Python `.py` files.
- No functions used to generate results are defined inside Jupyter notebooks.
- The illustrative notebook included in `jupyter_notebook/` exclusively loads precomputed outputs and calls the corresponding functions from `src/`.

This design ensures full transparency, avoids hidden execution logic, and allows direct inspection of every methodological step involved in the study.

---

## ⚠️ Scope and Limitations

This repository is intended to support **methodological transparency and reproducibility**, not real-time commercial deployment.

- Hyperparameters and model configurations correspond exactly to those used in the empirical analysis reported in the paper.
- The framework can be readily adapted to other assets or frequencies, but such extensions fall outside the scope of the present study.

---

## ✅ Compliance Summary

This repository complies with the software, data, and reproducibility requirements of *Energy Economics* by:

- Providing all in-house developed models,
- Using machine-readable data formats,
- Avoiding hidden logic in notebooks,
- Enabling independent verification of published results. 

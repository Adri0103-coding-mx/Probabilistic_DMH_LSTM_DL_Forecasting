# DMH-LSTM: A Real-Time Probabilistic Framework for Operational Crude Oil Price Forecasting

This repository provides the official Python implementation of the **DMH-LSTM** framework. This system is designed to bridge the gap between deep learning research and operational energy market requirements by ensuring a leakage-free, real-time forecasting pipeline.

# Project Overview
This repository hosts the code for the DMH-LSTM (Direct Multi-Horizon Long Short-Term Memory) framework, a robust deep learning system designed for the operational forecasting of crude oil prices. Unlike standard "offline" models that often suffer from look-ahead bias, this framework implements a strict real-time operational pipeline. It is specifically engineered to handle the non-linearities and volatility of energy markets, providing not just point forecasts, but calibrated uncertainty intervals crucial for financial risk management.

## 🚀 Key Features

* **Direct Multi-Horizon (DMH):** A forecasting strategy that prevents error propagation by training specific heads for each lead time.
* **Operational Pipeline:** Strict walk-forward backtesting methodology to eliminate look-ahead bias.
* **Calibrated Uncertainty:** Integration of **Conformal Prediction** to provide reliable prediction intervals during high-volatility events.
* **Model Interpretability:** Gradient-based **Saliency Maps** to visualize the impact of temporal lags on price direction.
* **Financial Risk Assessment:** Evaluation beyond standard errors using **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**.

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/DMH-LSTM-Oil-Forecasting.git](https://github.com/yourusername/DMH-LSTM-Oil-Forecasting.git)
cd DMH-LSTM-Oil-Forecasting

# Install dependencies
pip install -r requirements.txt

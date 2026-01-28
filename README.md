# Probabilistic_DMH_LSTM_DL_Forecasting
Official implementation of the DMH-LSTM framework for operational crude oil price forecasting. Features real-time walk-forward backtesting, Conformal Prediction for calibrated uncertainty, and gradient-based interpretability.

# Project Overview
This repository hosts the code for the DMH-LSTM (Direct Multi-Horizon Long Short-Term Memory) framework, a robust deep learning system designed for the operational forecasting of crude oil prices. Unlike standard "offline" models that often suffer from look-ahead bias, this framework implements a strict real-time operational pipeline. It is specifically engineered to handle the non-linearities and volatility of energy markets, providing not just point forecasts, but calibrated uncertainty intervals crucial for financial risk management.

# 🛠 Key Features
Direct Multi-Horizon (DMH) Strategy: Utilizes a direct forecasting approach to eliminate the error accumulation found in recursive strategies, providing consistent accuracy across multiple lead times.
Leakage-Free Pipeline: A rigorous walk-forward backtesting engine ensures that the model only uses information available at the time of the forecast, mirroring real-world trading conditions.
Calibrated Uncertainty: Employs Conformal Prediction (CP) to generate prediction intervals with theoretical coverage guarantees, even during extreme geopolitical shocks.
Model Interpretability (XAI): Includes gradient-based Saliency Maps to visualize the relevance of specific temporal lags, transforming the "black box" LSTM into an interpretable tool for economists.
Financial Risk Metrics: Evaluation goes beyond RMSE/MAE, incorporating Value-at-Risk (VaR), Expected Shortfall (ES), and QLIKE loss functions for a comprehensive assessment of forecasting utility.

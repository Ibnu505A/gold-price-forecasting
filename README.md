# Gold Price Forecasting

Predicting Gold/USD closing prices using an LSTM neural network trained on 25+ years of daily historical data.

## Results

| Metric | Value |
|--------|-------|
| MAE | $43.51 |
| RMSE | $59.02 |
| MAPE | **1.75%** |

## Dataset

`GoldUSD.csv` — daily OHLCV data from 2000 to 2026. Target variable: `Close` price.

## Approach

1. **EDA** — inspect data types, missing values, descriptive statistics
2. **Outlier handling** — IQR clipping computed on training data only (no leakage)
3. **Train/test split** — 80/20 chronological split
4. **Scaling** — StandardScaler fitted on training set, applied to both sets
5. **Sliding window** — window size of 30 days to create LSTM input sequences
6. **Model** — single LSTM layer (64 units) + Dropout (0.3) + Dense output
7. **Training** — Early Stopping (patience=5) with best weight restoration
8. **Forecasting** — recursive 30-day ahead prediction

## Stack

- Python 3.12
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Usage

Open `forecasting_fixed.ipynb` in Jupyter and run all cells. The notebook is self-contained and outputs all plots and metrics inline.

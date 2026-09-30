# Stock Price Prediction, Sales Forecasting and Waiter Tips Regression
## Author: Rabea Akter Shoily (Student ID 905253001) · United International University, School of Business and Economics
This project builds and compares predictive models for three problems on a synthetic dataset: an LSTM neural network for stock prices, Holt-Winters and SARIMA models for store sales, and regression models (baseline, Ridge, Random Forest, Gradient Boosting) for waiter tips. Every model is tested on data it never saw during training, using a time-based split for the stock and sales series. The best tips pipeline (Ridge) is saved to a single file and reloaded to predict new bills. The written report is included in the `report/` folder.
Repository structure
```
.
├── data/        predictive\_analytics\_dataset.xlsx   (3 sheets, see below)
├── notebooks/   predictive\_analytics\_project.ipynb  (full workflow, run top to bottom)
├── models/      tips\_pipeline.joblib                (saved preprocessing + Ridge model)
│                lstm\_retailco.keras, lstm\_scaler.joblib   (saved LSTM and its scaler)
├── outputs/     charts (01\_ to 07\_\*.png) and comparison tables (\*.csv)
├── report/      project report (Word)
├── requirements.txt
└── README.md
```
How to install and run
Install Python 3.10 or newer (developed with 3.12).
Install the dependencies:
```bash
   pip install -r requirements.txt
   ```
Open the notebook and run all cells from top to bottom:
```bash
   jupyter notebook notebooks/predictive\_analytics\_project.ipynb
   ```
The notebook reads `../data/predictive\_analytics\_dataset.xlsx` and writes charts to `../outputs` and saved models to `../models`, so keep the folder layout above and start Jupyter from the repository root or the `notebooks/` folder.
A fixed random seed (42) is used, so results should be reproducible.
Datasets (sheets in `data/predictive\_analytics\_dataset.xlsx`)
Sheet	Content	Used for
`Stock\_Prices`	Three tickers, about 750 trading days each, open/high/low/close and volume	LSTM on RETAILCO closing prices
`Sales\_Data`	Four stores, 730 daily sales figures with trend and weekly seasonality	Holt-Winters and SARIMA on the East store
`Waiter\_Tips`	300 restaurant bills with tip, party size, day, time, smoker, sex	Regression of `tip`
The data is synthetic and supplied by the instructor.
Key results
Stock (RETAILCO, last 150 days)
Model	RMSE
LSTM (30-day window)	2.744
Naive (previous close)	1.624
The LSTM did not beat the naive benchmark: test prices went above the highest training price, which a network trained on scaled prices handles poorly.
Sales (East store, last 60 days)
Model	RMSE
Seasonal naive	60.31
Holt-Winters	24.07
SARIMA(1,1,1)(1,0,1,7)	44.27
Holt-Winters is close to the noise floor (residual std about 23).
Waiter tips
Model	Train R²	Test R²	Test RMSE	CV R²
1. Baseline (bill only)	0.463	0.503	1.024	0.464
2. Linear (all features)	0.534	0.544	0.981	0.506
3. Ridge (tuned), deployed	0.523	0.539	0.986	0.506
4. Random Forest (tuned)	0.594	0.507	1.020	0.447
5. Gradient Boosting (tuned)	0.597	0.524	1.003	0.465
Refinements improved test R² by only about 0.04 because the total bill dominates tip size. Ridge was deployed: it matches the best cross-validated R², shows no overfitting, and is easy to explain.
Using the saved tips pipeline
```python
import joblib, pandas as pd
pipe = joblib.load("models/tips\_pipeline.joblib")
new = pd.DataFrame(\[{"total\_bill": 45.0, "sex": "Male", "smoker": "No",
                     "day": "Sat", "time": "Dinner", "size": 4}])
print(pipe.predict(new))   # about 6.81
```

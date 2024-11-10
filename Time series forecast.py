# Required Libraries
import warnings
import itertools
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Load dataset
data_url = "https://raw.githubusercontent.com/kirenz/datasets/master/IPG2211A2N.csv"
data = pd.read_csv(data_url)

# Convert date column to datetime and set as index
data.index = pd.to_datetime(data.DATE)
data = data.drop(['DATE'], axis=1)
data.columns = ['Production']

# Plotting the Production data
data.plot(figsize=(20, 10), linewidth=2, fontsize=16)
plt.xlabel('Year', fontsize=18)
plt.show()

# Decomposition of the time series (additive and multiplicative)
decomp_additive = seasonal_decompose(data, model='additive')
fig_additive = decomp_additive.plot()
fig_additive.set_size_inches(15, 8)

decomp_multiplicative = seasonal_decompose(data, model='multiplicative')
fig_multiplicative = decomp_multiplicative.plot()
fig_multiplicative.set_size_inches(15, 8)

# Perform Augmented Dickey-Fuller test
adf_result = adfuller(data['Production'])
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value:.3f}')

# Plot differenced data
data.diff().plot(figsize=(20, 10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.show()

# Select data subset for more detailed analysis
data_subset = data.iloc[850:900]
data_subset.diff().plot(figsize=(20, 5), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.show()

# Autocorrelation and Partial Autocorrelation
autocorrelation_plot(data)
plt.show()

autocorrelation_plot(data_subset)
plt.show()

plot_acf(data)
plt.show()

plot_acf(data_subset)
plt.show()

plot_pacf(data)
plt.show()

plot_pacf(data_subset)
plt.show()

# ARIMA Model Fitting
arima_model = ARIMA(data, order=(5, 1, 0))
arima_results = arima_model.fit()
print(arima_results.summary())

# Residuals analysis
residuals = pd.DataFrame(arima_results.resid)
residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()

print(residuals.describe())

# Define parameter ranges for SARIMA model
p = d = q = range(0, 2)
pdq_combinations = list(itertools.product(p, d, q))
seasonal_pdq_combinations = [(x[0], x[1], x[2], 12) for x in pdq_combinations]

print('Examples of SARIMA parameter combinations:')
for i in range(1, 4):
    print(f'SARIMAX: {pdq_combinations[i]} x {seasonal_pdq_combinations[i]}')

# Suppress warnings
warnings.filterwarnings("ignore")

# Search for optimal SARIMA parameters
for param in pdq_combinations:
    for param_seasonal in seasonal_pdq_combinations:
        try:
            sarima_model = sm.tsa.statespace.SARIMAX(
                data, order=param, seasonal_order=param_seasonal,
                enforce_stationarity=False, enforce_invertibility=False
            )
            sarima_results = sarima_model.fit()
            print(f'SARIMA{param}x{param_seasonal}12 - AIC:{sarima_results.aic}')
        except:
            continue

# Fit SARIMA model with chosen parameters
chosen_model = sm.tsa.statespace.SARIMAX(
    data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False, enforce_invertibility=False
)
final_results = chosen_model.fit()

print(final_results.summary().tables[1])

# Diagnostics of the final model
final_results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Forecasting
pred = final_results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_conf_int = pred.conf_int()

# Plot forecast
ax = data['2013':].plot(label='Observed')
pred.predicted_mean.plot(ax=ax, label='One-step Forecast', alpha=0.7)

ax.fill_between(pred_conf_int.index,
                pred_conf_int.iloc[:, 0],
                pred_conf_int.iloc[:, 1], color='k', alpha=0.2)
ax.set_xlabel('Date')
ax.set_ylabel('Production')
plt.legend()
plt.show()

# Calculate and print Mean Squared Error (MSE)
data['Forecasted'] = pred.predicted_mean
data['Squared_Error'] = (data['Production'] - pred.predicted_mean) ** 2
data_cleaned = data.dropna()
SS_R = data_cleaned['Squared_Error'].sum()
n_obs = len(data_cleaned["Squared_Error"])
mse = round((SS_R / (n_obs - 1)), 2)
print('Mean Squared Error:', mse)

data_cleaned.head(5)

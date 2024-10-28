import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('monthly_expenses_data.csv', parse_dates=['Date'], index_col='Date')

# Sidebar sliders for adjusting GDP and CPI values for the forecast period
st.sidebar.header("Adjust Forecast Parameters")
forecast_periods = st.sidebar.slider("Forecast Periods (Months)", min_value=1, max_value=24, value=12)
gdp_forecast = st.sidebar.slider("GDP Growth Rate (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
cpi_forecast = st.sidebar.slider("CPI Rate (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

# Prepare external regressors (GDP and CPI) for training and forecasting
external_regressors = data[['GDP_Growth_Rate', 'CPI_Rate']]

# Train the ARIMAX model
model = SARIMAX(data['Monthly_Expenses'], exog=external_regressors, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Create forecast values for the external regressors based on user input
future_gdp = [gdp_forecast] * forecast_periods
future_cpi = [cpi_forecast] * forecast_periods
future_exog = pd.DataFrame({'GDP_Growth_Rate': future_gdp, 'CPI_Rate': future_cpi})

# Generate forecast
forecast = model_fit.get_forecast(steps=forecast_periods, exog=future_exog)
forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot the results
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data['Monthly_Expenses'], label='Observed')
ax.plot(forecast_index, forecast_values, label='Forecast', color='orange')
ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
ax.set_title('Monthly Expenses Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Expenses')
ax.legend()

# Display the plot and model details in Streamlit
st.title("ARIMAX Model for Monthly Expenses Forecast")
st.write("This app uses an ARIMAX model to forecast monthly expenses, allowing users to adjust GDP and CPI forecasts.")
st.pyplot(fig)

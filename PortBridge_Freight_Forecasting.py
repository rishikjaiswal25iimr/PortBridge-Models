# ================================
# PORTBRIDGE FREIGHT FORECASTING
# Using Facebook Prophet (Google Colab)
# ================================

# Install Prophet
!pip install prophet --quiet

# Import Libraries
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("portbridge_freight_simulation.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Prepare data for Prophet
prophet_df = df[['Date', 'SCFI_Index']].rename(columns={
    'Date': 'ds',
    'SCFI_Index': 'y'
})

print("Data Preview:")
print(prophet_df.head())

# Initialize Prophet Model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

# Fit Model
model.fit(prophet_df)

# Create Future Dates (6 Months Forecast)
future = model.make_future_dataframe(periods=180)

# Generate Forecast
forecast = model.predict(future)

# ================================
# PLOT 1: Forecast with Confidence Interval
# ================================
fig1 = model.plot(forecast)
plt.title("PortBridge 6-Month Freight Rate Forecast")
plt.xlabel("Date")
plt.ylabel("Freight Rate Index")
plt.show()

# ================================
# PLOT 2: Trend + Seasonality Components
# ================================
fig2 = model.plot_components(forecast)
plt.show()

# ================================
# Display Last 10 Forecast Values
# ================================
print("\nLast 10 Forecasted Values:")
print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(10))

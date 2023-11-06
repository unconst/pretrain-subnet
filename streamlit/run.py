import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import wandb

# Initialize W&B API.
api = wandb.Api()

# Retrieve run data.
run = api.run("/opentensor-dev/openpretraining/runs/74iflokp")
history = run.history()

# Assuming 'best_average_loss' represents some kind of average loss per epoch, we'll use it for the candlestick.
# Also assuming 'best_average_loss_uid' could represent the "unique id" for the timeframe or epoch.

# Create a DataFrame for the candlestick chart.
# We need Open, High, Low, Close data for candlestick which we will assume:
# 'best_average_loss' will be used for Close
# 'best_average_loss_uid' for Open (as a placeholder, usually this would be the loss at the start of an epoch)
# We will use the min and max of the losses for Low and High respectively for each epoch
candlestick_data = pd.DataFrame({
    'Date': pd.to_datetime(history['_timestamp'], unit='s'),
    'Open': history['best_average_loss_uid'],  # Placeholder, this would typically be the loss at the beginning of an epoch
    'High': history[['223.loss', '215.loss', '214.loss']].max(axis=1),
    'Low': history[['223.loss', '215.loss', '214.loss']].min(axis=1),
    'Close': history['best_average_loss']
})

# Plotly candlestick chart.
fig = go.Figure(data=[go.Candlestick(x=candlestick_data['Date'],
                open=candlestick_data['Open'],
                high=candlestick_data['High'],
                low=candlestick_data['Low'],
                close=candlestick_data['Close'])])

# Streamlit layout.
st.title('Best Average Loss Timeseries')
st.plotly_chart(fig)

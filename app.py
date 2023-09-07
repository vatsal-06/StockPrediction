# Import libraries
import yfinance as yf
import streamlit as st
from datetime import date

from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly

# Set Title
st.title('Stock Prediction')


# Clear cache
@st.cache
# Ticker Setup
def get_ticker(name):
    company = yf.get_ticker(name)  # Get ticker data
    return company  # Return company name


# Load Data Function
def load_data(name, start, end):
    data = yf.download(name, start=start, end=end)  # Download data
    data.reset_index(inplace=True)
    return data


# Raw Data Plotting Function
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Prediction Function
def predict(data, days):
    # Set model parameters
    model = Prophet()

    # Fit collected Data & Set Prediction Period
    model.fit(data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}))
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Display forecast data
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {days} days')
    fig1 = plot_plotly(model, forecast, xlabel='Date', ylabel='Close', figsize=(900, 600))
    st.plotly_chart(fig1)

    return forecast  # Return forecast/prediction


# Main Function
def main():
    # Get user input for stock symbol, start date, and end date
    name = st.sidebar.text_input('Enter stock symbol:')  # Get stock symbol
    start_date = st.sidebar.date_input('Select start date:')  # Get start date
    end_date = st.sidebar.date_input('Select end date:')  # Get end date
    
    st.text('''
        Commonly Used Stock Symbols:
        1. AAPL
        2. ABNB
        3. SNAP
        4. TSLA
        5. IBM
        6. AMZN
        7. WMT
        8. MSFT
        9. GOOG
        10. JPM
        11. JNJ
        12. ORCL
        13. KO
        14. MCD
        15. ADBE
        16. NFLX
    ''')

    # Load data and make predictions
    if name:

        # Load data
        data_load_state = st.sidebar.text('Loading data...')
        data = load_data(name, start_date, end_date)
        data_load_state.text('Done!')

        if not data.empty:
            # Slider for number of days to predict
            # Change 5 to 365 to predict 1 year
            years = st.sidebar.slider('Select number of years to predict:', 1, 5)
            days = years * 365

            # Plot raw data
            st.subheader('Raw data')
            plot_raw_data(data)

            # Make prediction
            forecast = predict(data, days)


# Run Main Function
if __name__ == '__main__':
    main()

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Streamlit app configuration
st.set_page_config(page_title="Jaideep's Advanced Stock Market Analysis", layout="wide")
st.title("📈 Advanced Stock Market Analysis Web App")

# Sidebar for user inputs
st.sidebar.header("Stock Analysis Settings")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., 'AAPL', 'RELIANCE.NS')", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Function to compute RSI
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to compute Bollinger Bands
def compute_bollinger_bands(series, window=20, std_dev=2):
    sma = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return upper_band, lower_band

# Function to fetch stock data
@st.cache_data
def fetch_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to plot stock chart
def plot_stock_chart(data, symbol):
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="OHLC"
    ))

    # Add Moving Averages
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SMA_50'], line=dict(color='red', width=2), name="50-Day SMA"
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SMA_200'], line=dict(color='blue', width=2), name="200-Day SMA"
    ))

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Upper_Band'], line=dict(color='orange', width=1), name="Upper Bollinger Band"
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Lower_Band'], line=dict(color='orange', width=1), name="Lower Bollinger Band"
    ))

    # Layout
    fig.update_layout(
        title=f"{symbol} Stock Price Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    return fig

# Function to predict next day's price
def predict_next_price(data):
    from sklearn.linear_model import LinearRegression
    X = pd.Series(data.index).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = data['Close'].values
    model = LinearRegression().fit(X, y)
    next_day = pd.Timestamp(data.index[-1]) + pd.Timedelta(days=1)
    next_day_price = model.predict([[next_day.toordinal()]])[0]
    return next_day_price

# Load stock data
st.write(f"Fetching data for {symbol} from {start_date} to {end_date}...")
data = fetch_data(symbol, start_date, end_date)

if data is None:
    st.error("No data available for the given symbol and date range.")
else:
    # Display stock data
    st.subheader(f"📊 Stock Data for {symbol}")
    st.dataframe(data.tail(), height=250)

    # Compute Technical Indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['Upper_Band'], data['Lower_Band'] = compute_bollinger_bands(data['Close'])

    # Plot OHLC Chart with Moving Averages and Bollinger Bands
    fig = plot_stock_chart(data, symbol)
    st.plotly_chart(fig, use_container_width=True)

    # Display Statistics
    st.subheader(f"📈 Statistics for {symbol}")
    st.write(data.describe())

    # Machine Learning Prediction (Linear Regression for next day's price)
    st.subheader("🔮 Stock Price Prediction")
    next_day_price = predict_next_price(data)
    st.write(f"Predicted next day's closing price for {symbol}: ₹{float(next_day_price):.2f}")

    # Downloadable Data
    st.download_button("Download Processed Data", data.to_csv(), file_name=f"{symbol}_processed.csv", mime="text/csv")

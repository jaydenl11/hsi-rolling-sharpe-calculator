import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

def fetch_t_bill_yield(api_key):
    """
    Fetch the current yield of the US 3-month T-bill from FRED API
    :param api_key: Your FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
    :return: Annualized yield (decimal) or None if unsuccessful
    """
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&api_key={api_key}&file_type=json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                # Get most recent non-empty observation
                latest = next((obs for obs in reversed(data['observations']) 
                             if obs['value'] != '.'), None)
                if latest:
                    return float(latest['value']) / 100  # Convert percentage to decimal
        print(f"Warning: FRED API returned status {response.status_code}. Using fallback value.")
        return None
    except Exception as e:
        print(f"Error fetching from FRED: {str(e)}")
        return None

def calculate_rolling_sharpe(ticker='^HSI', window_days=21, api_key=None):
    """
    Calculate rolling Sharpe ratio using FRED for risk-free rate
    :param ticker: Stock/index ticker (default: ^HSI)
    :param window_days: Rolling window in trading days (default: 21)
    :param api_key: FRED API key (if None, uses default 0.02)
    """
    # Fetch or estimate risk-free rate
    annual_rf = fetch_t_bill_yield(api_key) if api_key else None
    if annual_rf is None:
        print("Using fallback risk-free rate: 2%")
        annual_rf = 0.02
    
    # Get asset data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=window_days * 3)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        print("Error: No price data fetched")
        return None

    # Prepare calculations
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    daily_rf = (1 + annual_rf) ** (1/252) - 1  # Annual to daily rate
    
    data['Daily Return'] = data[price_col].pct_change()
    data['Daily Risk Free'] = daily_rf
    
    # Rolling statistics
    data['Rolling Mean'] = data['Daily Return'].rolling(window_days).mean()
    data['Rolling Std'] = data['Daily Return'].rolling(window_days).std()
    
    # Annualized Sharpe ratio
    data['Rolling Sharpe'] = ((data['Rolling Mean'] - data['Daily Risk Free']) / 
                             data['Rolling Std'] * np.sqrt(252))
    
    return data[data['Rolling Sharpe'].notna()].copy()

def display_results(data, ticker):
    """Format and print results"""
    if data is None:
        return
        
    results = data[['Close', 'Daily Return', 'Daily Risk Free', 
                   'Rolling Mean', 'Rolling Std', 'Rolling Sharpe']].tail(10)
    
    print(f"\n{ticker} Rolling Sharpe Components (Latest 10 Days):")
    print("=" * 70)
    with pd.option_context('display.float_format', '{:.6f}'.format):
        print(results)
        
    print("\nSummary Statistics:")
    print(f"- Annual Risk-Free Rate: {100 * data['Daily Risk Free'].iloc[-1] * 252:.2f}%")
    print(f"- Latest Sharpe Ratio: {data['Rolling Sharpe'].iloc[-1]:.4f}")
    print(f"- Volatility (Std Dev): {data['Rolling Std'].iloc[-1] * 100:.2f}%")

def plot_results(data, ticker):
    """Visualize results"""
    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Sharpe Ratio
    ax[0].plot(data.index, data['Rolling Sharpe'], label='Sharpe Ratio', color='navy')
    ax[0].set_ylabel('Sharpe Ratio')
    ax[0].legend()
    ax[0].grid(True)
    
    # Returns
    ax[1].plot(data.index, data['Rolling Mean'] * 100, 
              label='Avg Daily Return (%)', color='green')
    ax[1].plot(data.index, data['Daily Risk Free'] * 100, 
              label='Risk-Free Rate (%)', color='red', linestyle='--')
    ax[1].set_ylabel('Returns (%)')
    ax[1].legend()
    ax[1].grid(True)
    
    # Volatility
    ax[2].plot(data.index, data['Rolling Std'] * 100, 
              label='Volatility (%)', color='purple')
    ax[2].set_ylabel('Std Dev (%)')
    ax[2].legend()
    ax[2].grid(True)
    
    plt.suptitle(f"{ticker} {len(data['Rolling Sharpe'])}-Day Rolling Analysis")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configuration
    TICKER = '^HSI'          # Hang Seng Index
    WINDOW = 21              # 1 month trading days
    FRED_API_KEY = '39ff98508a19323af327820d1b85ce0f'      # Replace with your actual key
    
    # Run analysis
    print(f"Calculating {WINDOW}-day rolling Sharpe for {TICKER}...")
    df = calculate_rolling_sharpe(TICKER, WINDOW, FRED_API_KEY)
    
    if df is not None:
        display_results(df, TICKER)
        plot_results(df, TICKER)
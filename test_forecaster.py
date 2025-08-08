#!/usr/bin/env python3
"""
Test script for the S&P 500 forecaster
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sp500_forecaster import forecaster
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import time


def test_single_stock():
    """Test forecasting for a single stock"""
    print("Testing single stock forecast...")

    # Test with AAPL
    ticker = "AAPL"

    # Prepare data
    train_data, test_data, scaler = forecaster.prepare_data(ticker)

    if train_data is None:
        print(f"❌ Failed to prepare data for {ticker}")
        return False

    print(f"✅ Data prepared for {ticker}")

    # Train model
    model, scaler = forecaster.train_model(ticker, train_data, test_data, scaler)

    if model is None:
        print(f"❌ Failed to train model for {ticker}")
        return False

    print(f"✅ Model trained for {ticker}")

    # Generate forecast
    forecast = forecaster.generate_forecast(ticker, model, scaler)

    if forecast is None:
        print(f"❌ Failed to generate forecast for {ticker}")
        return False

    print(f"✅ Forecast generated for {ticker}")
    print(f"   Next day prediction: ${forecast.iloc[0]['Predicted_Close']:.2f}")

    return True


def test_current_prices():
    """Test getting current prices"""
    print("\nTesting current price retrieval...")

    tickers = ["AAPL", "MSFT", "GOOGL"]

    for ticker in tickers:
        try:
            data = yf.download(ticker, period="1d")
            if not data.empty:
                price = float(data["Close"].iloc[-1])
                print(f"✅ {ticker}: ${price:.2f}")
            else:
                print(f"❌ {ticker}: No data available")
        except Exception as e:
            print(f"❌ {ticker}: Error - {e}")


def test_forecaster_initialization():
    """Test forecaster initialization"""
    print("\nTesting forecaster initialization...")

    try:
        # Check if directories are created
        for dir_path in [
            forecaster.data_dir,
            forecaster.models_dir,
            forecaster.predictions_dir,
            forecaster.scalers_dir,
        ]:
            if os.path.exists(dir_path):
                print(f"✅ Directory exists: {dir_path}")
            else:
                print(f"❌ Directory missing: {dir_path}")

        # Check ticker list
        print(f"✅ S&P 500 tickers loaded: {len(forecaster.SP500_TICKERS)} stocks")

        return True
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False


def test_price_cache():
    """Test the price caching mechanism"""
    print("\nTesting price cache (TTL)...")
    tickers = ["AAPL", "MSFT"]
    t0 = time.time()
    prices1 = forecaster.get_intraday_prices(tickers)
    t1 = time.time()
    prices2 = forecaster.get_intraday_prices(tickers)
    t2 = time.time()
    print(
        f"First fetch duration: {t1 - t0:.2f}s, second fetch: {t2 - t1:.2f}s (should be faster due to cache)"
    )
    print("Prices1:", prices1)
    print("Prices2:", prices2)


def test_concurrent_updates():
    """Test the guarding against concurrent updates"""
    print("\nTesting concurrent update guard...")

    def run_update():
        return forecaster.update_all_forecasts()

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(run_update) for _ in range(3)]
        results = [f.result() for f in futures]

    success_counts = [len(r) for r in results]
    print("Update result sizes (expect one non-zero, others zero):", success_counts)


def main():
    print("Running S&P 500 Forecaster Tests...")
    print("=" * 50)

    # Test initialization
    if not test_forecaster_initialization():
        print("❌ Initialization test failed")
        return

    # Test current prices
    test_current_prices()

    # Test single stock forecast
    if test_single_stock():
        print("\n✅ All tests passed! The forecaster is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python generate_forecasts.py' to generate all forecasts")
        print("2. Run 'python main.py' to start the Flask app")
        print("3. Visit http://localhost:5000/SP500 to view forecasts")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

    # Additional tests
    test_price_cache()
    test_concurrent_updates()


if __name__ == "__main__":
    main()

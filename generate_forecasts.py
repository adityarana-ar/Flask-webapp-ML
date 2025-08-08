#!/usr/bin/env python3
"""
Script to manually generate S&P 500 forecasts
Run this script to generate initial forecasts for all S&P 500 stocks
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sp500_forecaster import forecaster
import time


def main():
    print("Starting S&P 500 forecast generation...")
    print("This will take some time as we process 100 stocks...")

    start_time = time.time()

    try:
        # Generate all forecasts
        successful_forecasts = forecaster.update_all_forecasts()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nForecast generation completed!")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Successful forecasts: {len(successful_forecasts)}")
        print(
            f"Failed forecasts: {len(forecaster.SP500_TICKERS) - len(successful_forecasts)}"
        )

        if successful_forecasts:
            print("\nSample successful forecasts:")
            for i, (ticker, data) in enumerate(list(successful_forecasts.items())[:5]):
                print(f"  {ticker}: ${data['next_day_prediction']:.2f}")

        print(f"\nForecasts saved to: {forecaster.data_dir}")
        print("You can now run the Flask app and visit /SP500 to view the forecasts.")

    except Exception as e:
        print(f"Error during forecast generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

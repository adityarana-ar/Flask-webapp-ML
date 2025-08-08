import os
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Activation
import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import pickle
import threading
import time
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import contextlib

# S&P 500 tickers (top 100 for demonstration - you can expand this)
SP500_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "BRK-B",
    "LLY",
    "TSLA",
    "UNH",
    "JNJ",
    "JPM",
    "V",
    "PG",
    "XOM",
    "HD",
    "CVX",
    "MA",
    "PFE",
    "ABBV",
    "KO",
    "PEP",
    "COST",
    "TMO",
    "DHR",
    "ACN",
    "WMT",
    "NEE",
    "MRK",
    "VZ",
    "CMCSA",
    "ADBE",
    "PM",
    "TXN",
    "UNP",
    "RTX",
    "NFLX",
    "QCOM",
    "HON",
    "LOW",
    "INTC",
    "IBM",
    "MS",
    "GS",
    "CAT",
    "AMAT",
    "SPGI",
    "INTU",
    "VRTX",
    "ISRG",
    "GILD",
    "T",
    "ADI",
    "REGN",
    "PLD",
    "AMGN",
    "SYK",
    "MDLZ",
    "TMUS",
    "ADP",
    "ZTS",
    "KLAC",
    "BDX",
    "MU",
    "SO",
    "DUK",
    "NSC",
    "CME",
    "ITW",
    "TJX",
    "USB",
    "SHW",
    "ICE",
    "MMC",
    "AXP",
    "MO",
    "CCI",
    "TGT",
    "EOG",
    "AON",
    "SLB",
    "PGR",
    "NOC",
    "ETN",
    "ORCL",
    "COP",
    "PSA",
    "FISV",
    "APD",
    "SRE",
    "VLO",
    "CSCO",
    "GE",
    "AIG",
]


class SP500Forecaster:
    def __init__(self, data_dir="forecast_data"):
        self.data_dir = data_dir
        self.models_dir = os.path.join(data_dir, "models")
        self.predictions_dir = os.path.join(data_dir, "predictions")
        self.scalers_dir = os.path.join(data_dir, "scalers")

        # Make SP500_TICKERS accessible as class attribute
        self.SP500_TICKERS = SP500_TICKERS

        # Create directories if they don't exist
        for dir_path in [
            self.data_dir,
            self.models_dir,
            self.predictions_dir,
            self.scalers_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

        self.last_update_file = os.path.join(data_dir, "last_update.json")
        self.forecasts_cache = {}
        self._price_cache = {}
        self._price_cache_ts = None
        self._price_cache_ttl = 300  # seconds
        self._update_lock = threading.Lock()
        self._update_in_progress = False
        self.load_last_update()

    def load_last_update(self):
        """Load the last update timestamp"""
        if os.path.exists(self.last_update_file):
            with open(self.last_update_file, "r") as f:
                data = json.load(f)
                self.last_update = datetime.fromisoformat(data["last_update"])
        else:
            self.last_update = None

    def save_last_update(self):
        """Save the current update timestamp"""
        with open(self.last_update_file, "w") as f:
            json.dump({"last_update": datetime.now().isoformat()}, f)

    def compute_rsi(self, series, period=15):
        """Compute RSI indicator"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_data(self, ticker):
        """Prepare data for a specific ticker"""
        try:
            # Download data
            df = yf.download(tickers=ticker, period="3y")

            if df.empty:
                return None, None, None

            # Add technical indicators
            df["EMAF"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMAM"] = df["Close"].ewm(span=60, adjust=False).mean()
            df["EMAS"] = df["Close"].ewm(span=100, adjust=False).mean()
            df["RSI"] = self.compute_rsi(df["Close"], period=15)
            df["Target"] = df["Close"].shift(-1)

            df = df.dropna()

            if len(df) < 100:  # Need sufficient data
                return None, None, None

            # Prepare features
            cols = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "EMAF",
                "EMAM",
                "EMAS",
                "RSI",
                "Target",
            ]
            df = df[cols].astype(float)

            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled = scaler.fit_transform(df)

            # Create sequences
            pastcandles = 30
            num_features = df.shape[1]

            X = []
            for i in range(num_features):
                X.append([])
                for j in range(pastcandles, df_scaled.shape[0]):
                    X[i].append(df_scaled[j - pastcandles : j, i])

            X = np.moveaxis(X, [0], [2])
            X = np.array(X)
            Y = np.array(df_scaled[pastcandles:, -1])
            Y = np.reshape(Y, (len(Y), 1))

            # Split data
            ratio = int(len(X) * 0.8)
            X_train, X_test = X[:ratio], X[ratio:]
            Y_train, Y_test = Y[:ratio], Y[ratio:]

            return (X_train, Y_train), (X_test, Y_test), scaler

        except Exception as e:
            print(f"Error preparing data for {ticker}: {e}")
            return None, None, None

    def create_model(self, num_features):
        """Create LSTM model"""
        lstm = Input(shape=(30, num_features), name="LSTM_input")
        lstm_input = LSTM(150, name="First_layer")(lstm)
        lstm_input = Dense(1, name="Dense_layer")(lstm_input)
        result = Activation("linear", name="Result")(lstm_input)
        model = Model(inputs=lstm, outputs=result)

        adam = optimizers.Adam()
        model.compile(optimizer=adam, loss="mse")
        return model

    def train_model(self, ticker, train_data, test_data, scaler):
        """Train model for a specific ticker"""
        try:
            X_train, Y_train = train_data
            X_test, Y_test = test_data

            model = self.create_model(X_train.shape[2])

            # Train model
            model.fit(
                x=X_train,
                y=Y_train,
                batch_size=15,
                epochs=30,
                shuffle=True,
                validation_split=0.1,
                verbose=0,
            )

            # Save model and scaler
            model_path = os.path.join(self.models_dir, f"{ticker}_model.h5")
            scaler_path = os.path.join(self.scalers_dir, f"{ticker}_scaler.pkl")

            model.save(model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            return model, scaler

        except Exception as e:
            print(f"Error training model for {ticker}: {e}")
            return None, None

    def generate_forecast(self, ticker, model, scaler, days_ahead=10):
        """Generate forecast for a ticker"""
        try:
            # Get latest data for prediction
            df = yf.download(tickers=ticker, period="60d")

            if df.empty:
                return None

            # Prepare latest data
            df["EMAF"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMAM"] = df["Close"].ewm(span=60, adjust=False).mean()
            df["EMAS"] = df["Close"].ewm(span=100, adjust=False).mean()
            df["RSI"] = self.compute_rsi(df["Close"], period=15)

            # Use the same columns as training (including Target column)
            cols = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "EMAF",
                "EMAM",
                "EMAS",
                "RSI",
                "Target",  # ← Add Target column
            ]

            # Add Target column (we'll fill it with Close prices initially)
            df["Target"] = df["Close"].shift(-1)
            df = df.dropna()

            df = df[cols].astype(float)

            # Scale data
            df_scaled = scaler.transform(df)

            # Get last 30 days for prediction (now with 10 features)
            last_30_days = df_scaled[-30:].reshape(1, 30, len(cols))

            # Generate predictions
            predictions = []
            current_input = last_30_days.copy()

            for _ in range(days_ahead):
                pred = model.predict(current_input, verbose=0)
                predictions.append(pred[0, 0])

                # Update input for next prediction
                new_row = current_input[0, -1].copy()
                new_row[-1] = pred[0, 0]  # Update target
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1] = new_row

            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions_full = np.zeros((len(predictions), len(cols)))
            predictions_full[:, -1] = predictions.flatten()
            predictions_inv = scaler.inverse_transform(predictions_full)[:, -1]

            # Generate future dates
            us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            last_date = df.index[-1]
            future_dates = pd.date_range(last_date, periods=days_ahead + 1, freq=us_bd)[
                1:
            ]

            return pd.DataFrame(
                {"Date": future_dates, "Predicted_Close": predictions_inv}
            )

        except Exception as e:
            print(f"Error generating forecast for {ticker}: {e}")
            return None

    def _download_with_retry(
        self, tickers, period="1d", group_by="ticker", attempts=3, sleep_seconds=2
    ):
        last_err = None
        for i in range(attempts):
            try:
                data = yf.download(
                    tickers=tickers, period=period, group_by=group_by, threads=False
                )
                if data is not None and not getattr(data, "empty", False):
                    return data
            except Exception as e:
                last_err = e
            time.sleep(sleep_seconds * (i + 1))
        if last_err:
            print(f"Price download failed after {attempts} attempts: {last_err}")
        return None

    def get_intraday_prices(self, tickers):
        """Return current (latest daily) prices with simple in-memory TTL cache."""
        now = time.time()
        # If cache fresh and contains all requested tickers
        if (
            self._price_cache_ts
            and (now - self._price_cache_ts) < self._price_cache_ttl
            and all(t in self._price_cache for t in tickers)
        ):
            return {t: self._price_cache.get(t) for t in tickers}

        data = self._download_with_retry(tickers=tickers)
        prices = {}
        if data is not None:
            for t in tickers:
                try:
                    if t in data:  # multi-indexed columns
                        df_t = data[t]
                        if not df_t.empty and "Close" in df_t.columns:
                            prices[t] = float(df_t["Close"].iloc[0])
                    elif (
                        hasattr(data, "columns")
                        and "Close" in data.columns
                        and len(tickers) == 1
                    ):
                        prices[t] = float(data["Close"].iloc[0])
                except Exception as e:
                    print(f"Price extraction error {t}: {e}")
        # Update cache only if we retrieved something (avoid overwriting with empty)
        if prices:
            self._price_cache.update(prices)
            self._price_cache_ts = now
        return {t: prices.get(t) for t in tickers}

    def update_all_forecasts(self):
        """Update forecasts for all S&P 500 stocks (thread-safe, lock guarded)."""
        with self._update_lock:  # thread-safe context manager
            print(f"Starting forecast update for {len(self.SP500_TICKERS)} stocks...")

            successful_forecasts = {}

            for i, ticker in enumerate(self.SP500_TICKERS):
                print(f"Processing {ticker} ({i+1}/{len(self.SP500_TICKERS)})")

                try:
                    # Prepare data
                    train_data, test_data, scaler = self.prepare_data(ticker)

                    if train_data is None:
                        print(f"Skipping {ticker} - insufficient data")
                        continue

                    # Train model
                    model, scaler = self.train_model(
                        ticker, train_data, test_data, scaler
                    )

                    if model is None:
                        continue

                    # Generate forecast
                    forecast = self.generate_forecast(ticker, model, scaler)

                    if forecast is not None:
                        # Save forecast
                        forecast_path = os.path.join(
                            self.predictions_dir, f"{ticker}_forecast.csv"
                        )
                        forecast.to_csv(forecast_path, index=False)

                        successful_forecasts[ticker] = {
                            "last_updated": datetime.now().isoformat(),
                            "forecast_path": forecast_path,
                            "next_day_prediction": float(
                                forecast.iloc[0]["Predicted_Close"]
                            ),
                        }

                        print(f"✓ {ticker} forecast completed")
                    else:
                        print(f"✗ {ticker} forecast failed")

                except Exception as e:
                    print(f"✗ Error processing {ticker}: {e}")

            # Save summary
            summary_path = os.path.join(self.data_dir, "forecast_summary.json")
            with open(summary_path, "w") as f:
                json.dump(successful_forecasts, f, indent=2)

            self.save_last_update()
            print(
                f"Forecast update completed. {len(successful_forecasts)} successful forecasts."
            )
            return successful_forecasts

    def should_update(self):
        """Check if forecasts should be updated (daily after market close)"""
        if self.last_update is None:
            return True

        now = datetime.now()
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Update if it's after market close and we haven't updated today
        if now > market_close and self.last_update.date() < now.date():
            return True

        return False

    def get_forecast(self, ticker):
        """Get forecast for a specific ticker"""
        forecast_path = os.path.join(self.predictions_dir, f"{ticker}_forecast.csv")

        if os.path.exists(forecast_path):
            return pd.read_csv(forecast_path)
        else:
            return None

    def get_all_forecasts(self):
        """Get all available forecasts"""
        forecasts = {}
        for ticker in self.SP500_TICKERS:
            forecast = self.get_forecast(ticker)
            if forecast is not None:
                forecasts[ticker] = forecast
        return forecasts


# Global forecaster instance
forecaster = SP500Forecaster()


def background_update_forecasts():
    """Background thread to update forecasts"""
    while True:
        try:
            if forecaster.should_update():
                print("Starting background forecast update...")
                forecaster.update_all_forecasts()
                print("Background forecast update completed.")

            # Sleep for 1 hour before checking again
            time.sleep(3600)

        except Exception as e:
            print(f"Error in background forecast update: {e}")
            time.sleep(3600)


# Start background thread (guarded by environment variable)
if os.environ.get("ENABLE_BACKGROUND_UPDATES", "1") == "1":
    update_thread = threading.Thread(target=background_update_forecasts, daemon=True)
    update_thread.start()
else:
    print("Background updates disabled via ENABLE_BACKGROUND_UPDATES env var.")

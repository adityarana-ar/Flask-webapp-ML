import os
from flask import (
    Flask,
    render_template,
    request,
    url_for,
    flash,
    redirect,
    jsonify,
    Response,
    stream_template,
)

# Import the S&P 500 forecaster
from sp500_forecaster import forecaster

# Global variable to track active LSTM sessions
active_lstm_sessions = {}
import threading
import time
from flask_caching import Cache
from flask_cachecontrol import dont_cache

app = Flask(__name__)
img = os.path.join("static", "Image")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["SECRET_KEY"] = os.urandom(24).hex()

cache = Cache(
    app,
    config={
        "CACHE-TYPE": "RedisCache",
        "CACHE_REDIS_URL": "redis://127.0.0.1:6379/0",
        "CACHE_DEFAULT_TIMEOUT": 600,
    },
)


@app.route("/")
def home():
    return render_template("index.html")


d_input = [
    {
        "pregnancies": 0,
        "glucose": 0,
        "bp": 0,
        "skin_thickness": 0,
        "insulin": 0,
        "bmi": 0,
        "dpf": 0,
        "age": 0,
    }
]


@app.route("/Diabetes", methods=("GET", "POST"))
def diabetes():
    if request.method == "POST":
        pregnancies = request.form["pregnancies"]
        glucose = request.form["glucose"]
        bp = request.form["blood_pressure"]
        skin_thickness = request.form["skin_thickness"]
        insulin = request.form["insulin"]
        bmi = request.form["bmi"]
        dpf = request.form["dpf"]
        age = request.form["age"]

        if not pregnancies:
            flash("Please enter valid numbers")
        elif not glucose:
            flash("Please enter valid numbers")
        elif not bp:
            flash("Please enter valid numbers")
        elif not skin_thickness:
            flash("Please enter valid numbers")
        elif not insulin:
            flash("Please enter valid numbers")
        elif not bmi:
            flash("Please enter valid numbers")
        elif not dpf:
            flash("Please enter valid numbers")
        elif not age:
            flash("Please enter valid numbers")
        try:
            d_input.append(
                {
                    "pregnancies": pregnancies,
                    "glucose": glucose,
                    "bp": bp,
                    "skin_thickness": skin_thickness,
                    "insulin": insulin,
                    "bmi": bmi,
                    "dpf": dpf,
                    "age": age,
                }
            )

            import numpy as np  # To make numpy arrays
            import pandas as pd  # Used to create data frames (tables)
            from sklearn.preprocessing import StandardScaler  # To standardize the data
            from sklearn.model_selection import (
                train_test_split,
            )  # To split the data into

            # training and testing

            from sklearn import svm  # Importing Support Vector Machine model
            from sklearn.metrics import accuracy_score

            """Data Collection and Analysis"""

            # We can probably change the data or switch to a different kind of data.
            # this is from a tutorial just to see how well the model works.

            # Loading diabetes.csv through panda dataframe

            df = pd.read_csv("diabetes.csv")

            # Printing first 5 rows of dataset
            # df.head()

            # df.describe()

            # df['Pregnancies'].value_counts()

            # df['Outcome'].value_counts()

            # Separating data and labels
            X = df.drop(
                columns="Outcome", axis=1
            )  # column of outcome is dropped. rest is assigned
            Y = df["Outcome"]  # Only outcome is stored corresponding to df's indexes

            # print(X)

            # Standardizing data now since all values have differenct ranges.
            scaler = StandardScaler()

            standardized_data = scaler.fit_transform(
                X
            )  # Fitting and transforming X into
            # variable standardized_data
            # print(standardized_data)

            # Ranges are similar across different columns
            X = standardized_data
            Y = df["Outcome"]

            # print(Y)

            # Splitting data into Training data and test data
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, stratify=Y, random_state=2
            )
            # 4 variables here, X being split in train and test
            # 0.2 = 20% of data (how much of data is test data)
            # Stratify is to ensure that the X_train, X_test have the same proportion (diabestic, non-diabetic) as data
            # So it will eliminate all non-diabetic data going to train and it being tested on all diabetic people
            # Random state is to replicate the same splitting for testing purposes ig

            # print(X.shape, X_train.shape, X_test.shape)

            # 20% went to X_test rest 80% is with X_train (i.e. more date to train which is good)

            # TRAINING THE MODEL
            # Loading support vector machine
            classifier = svm.SVC(kernel="linear")

            # Training SVM
            classifier.fit(X_train, Y_train)

            # Evaluating the model
            # Accuracy score on the training data
            X_train_prediction = classifier.predict(X_train)
            training_accuracy = accuracy_score(X_train_prediction, Y_train)
            # print("Accuracy score of the training data: ", training_accuracy)

            X_test_prediction = classifier.predict(X_test)
            test_accuracy = accuracy_score(X_test_prediction, Y_test)
            print("Accuracy score of the test data: ", test_accuracy)

            # We have our model ready now we just need a predictive system

            input_data = (
                pregnancies,
                glucose,
                bp,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age,
            )

            # Change it to numpy array

            input_data = np.asarray(input_data)

            # Re-shaping the array (as model is expecting 768 values rn)

            input_data = input_data.reshape(1, -1)

            # Standardizing the data

            input_data = scaler.transform(input_data)

            # print(input_data)

            prediction = classifier.predict(input_data)
            # print(prediction)
            if prediction == 0:
                result = "non-diabetic"
            else:
                result = "diabetic"

            return render_template("Diabetes_Results.html", prediction=result)
        except:
            flash("An error ocurred. Try using valid parameters for the web form.")

    return render_template("Diabetes.html")


from flask_caching import Cache

cache = Cache(
    app,
    config={
        "CACHE-TYPE": "RedisCache",
        "CACHE_REDIS_URL": "redis://127.0.0.1:6379/0",
        "CACHE_DEFAULT_TIMEOUT": 600,
    },
)


@app.route("/SP500", methods=("GET", "POST"))
@dont_cache()
def sp500_forecasts():
    """Display S&P 500 stock forecasts with loading screen."""
    # Redirect to progress page first
    return redirect(url_for("sp500_progress"))


@app.route("/SP500/progress")
@dont_cache()
def sp500_progress():
    """Progress page for SP500 loading"""
    return render_template("SP500_Progress.html")


# Global storage for SP500 loading status
sp500_status = {}


@app.route("/SP500/stream")
@dont_cache()
def sp500_stream():
    """Streaming endpoint for SP500 loading"""

    def generate():
        try:
            from datetime import datetime  # local import to avoid global clutter

            # Step 1: Loading cached forecasts
            sp500_status["step"] = 1
            sp500_status["message"] = "Loading cached forecasts..."
            sp500_status["progress"] = 20
            yield 'data: {"step": 1, "message": "Loading cached forecasts...", "progress": 20}\n\n'

            # 1) Load precomputed forecasts from disk
            all_forecasts = forecaster.get_all_forecasts()
            if not all_forecasts:
                sp500_status["step"] = "error"
                sp500_status["message"] = (
                    "No cached forecasts found. They may still be generating in the background."
                )
                yield 'data: {"step": "error", "message": "No cached forecasts found. They may still be generating in the background."}\n\n'
                return

            # Step 2: Fetching current prices
            sp500_status["step"] = 2
            sp500_status["message"] = "Fetching current prices..."
            sp500_status["progress"] = 40
            yield 'data: {"step": 2, "message": "Fetching current prices...", "progress": 40}\n\n'

            # 2) Get current prices via your TTL cache (single shot for all tickers)
            tickers = list(all_forecasts.keys())
            prices = forecaster.get_intraday_prices(tickers) or {}

            # Step 3: Calculating price changes
            sp500_status["step"] = 3
            sp500_status["message"] = "Calculating price changes..."
            sp500_status["progress"] = 60
            yield 'data: {"step": 3, "message": "Calculating price changes...", "progress": 60}\n\n'

            forecasts_with_prices = {}
            total_forecasts = len(all_forecasts)
            last_updated_count = 0
            prediction_changes = []

            for ticker, forecast_df in all_forecasts.items():
                try:
                    next_day_prediction = (
                        float(forecast_df.iloc[0]["Predicted_Close"])
                        if not forecast_df.empty
                        else None
                    )

                    # Price from cached dict (may be None if unavailable)
                    price = prices.get(ticker)

                    change_percentage = None
                    if (
                        price is not None
                        and next_day_prediction is not None
                        and price != 0
                    ):
                        change_percentage = (
                            (next_day_prediction - price) / price
                        ) * 100.0
                        prediction_changes.append(change_percentage)

                    # File mtime → last updated stamp, and count today's updates
                    last_updated = None
                    forecast_path = os.path.join(
                        forecaster.predictions_dir, f"{ticker}_forecast.csv"
                    )
                    if os.path.exists(forecast_path):
                        file_time = datetime.fromtimestamp(
                            os.path.getmtime(forecast_path)
                        )
                        if file_time.date() == datetime.now().date():
                            last_updated_count += 1
                        last_updated = file_time.strftime("%Y-%m-%d %H:%M")

                    forecasts_with_prices[ticker] = {
                        "next_day_prediction": next_day_prediction,
                        "current_price": price,
                        "change_percentage": change_percentage,
                        "last_updated": last_updated,
                    }
                except Exception as inner_e:
                    print(f"Error processing {ticker}: {inner_e}")
                    continue

            # Step 4: Preparing display data
            sp500_status["step"] = 4
            sp500_status["message"] = "Preparing display data..."
            sp500_status["progress"] = 80
            yield 'data: {"step": 4, "message": "Preparing display data...", "progress": 80}\n\n'

            avg_prediction_change = (
                (sum(prediction_changes) / len(prediction_changes))
                if prediction_changes
                else 0
            )
            last_update = (
                forecaster.last_update.strftime("%Y-%m-%d %H:%M")
                if forecaster.last_update
                else None
            )

            # Step 5: Finalizing results
            sp500_status["step"] = 5
            sp500_status["message"] = "Finalizing results..."
            sp500_status["progress"] = 90
            yield 'data: {"step": 5, "message": "Finalizing results...", "progress": 90}\n\n'

            # Store results for later access
            sp500_status["results"] = {
                "forecasts": forecasts_with_prices,
                "total_forecasts": total_forecasts,
                "last_updated_count": last_updated_count,
                "avg_prediction_change": avg_prediction_change,
                "last_update": last_update,
            }

            # Send completion message
            sp500_status["step"] = "complete"
            sp500_status["message"] = "Loading complete! Redirecting to results..."
            sp500_status["progress"] = 100
            yield 'data: {"step": "complete", "message": "Loading complete! Redirecting to results...", "progress": 100, "redirect": "/SP500/results"}\n\n'

        except Exception as e:
            import traceback

            print("Exception in SP500 progress:", e)
            traceback.print_exc()
            sp500_status["step"] = "error"
            sp500_status["message"] = f"Error: {str(e)}"
            yield f'data: {{"step": "error", "message": "Error: {str(e)}"}}\n\n'

    return Response(generate(), mimetype="text/plain")


@app.route("/SP500/results")
@dont_cache()
def sp500_results():
    """Display SP500 results"""
    try:
        # Check if we have results from the streaming process
        if "results" not in sp500_status:
            # Fallback to direct loading if no streaming results
            return sp500_forecasts_direct()

        results = sp500_status["results"]

        return render_template(
            "SP500_Results.html",
            forecasts=results["forecasts"],
            total_forecasts=results["total_forecasts"],
            last_updated_count=results["last_updated_count"],
            avg_prediction_change=results["avg_prediction_change"],
            last_update=results["last_update"],
        )
    except Exception as e:
        print(f"Error in SP500 results: {e}")
        return render_template(
            "SP500_Results.html",
            error=f"Error displaying results: {str(e)}",
            forecasts={},
            total_forecasts=0,
            last_updated_count=0,
            avg_prediction_change=0,
            last_update=None,
        )


def sp500_forecasts_direct():
    """Direct SP500 forecasts loading (fallback method)"""
    from datetime import datetime  # local import to avoid global clutter

    try:
        # 1) Load precomputed forecasts from disk
        all_forecasts = forecaster.get_all_forecasts()
        if not all_forecasts:
            return render_template(
                "SP500_Results.html",
                forecasts={},
                total_forecasts=0,
                last_updated_count=0,
                avg_prediction_change=0,
                last_update=(
                    forecaster.last_update.strftime("%Y-%m-%d %H:%M")
                    if forecaster.last_update
                    else None
                ),
                error="No cached forecasts found. They may still be generating in the background.",
            )

        # 2) Get current prices via your TTL cache (single shot for all tickers)
        tickers = list(all_forecasts.keys())
        prices = forecaster.get_intraday_prices(tickers) or {}

        forecasts_with_prices = {}
        total_forecasts = len(all_forecasts)
        last_updated_count = 0
        prediction_changes = []

        for ticker, forecast_df in all_forecasts.items():
            try:
                next_day_prediction = (
                    float(forecast_df.iloc[0]["Predicted_Close"])
                    if not forecast_df.empty
                    else None
                )

                # Price from cached dict (may be None if unavailable)
                price = prices.get(ticker)

                change_percentage = None
                if price is not None and next_day_prediction is not None and price != 0:
                    change_percentage = ((next_day_prediction - price) / price) * 100.0
                    prediction_changes.append(change_percentage)

                # File mtime → last updated stamp, and count today's updates
                last_updated = None
                forecast_path = os.path.join(
                    forecaster.predictions_dir, f"{ticker}_forecast.csv"
                )
                if os.path.exists(forecast_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(forecast_path))
                    if file_time.date() == datetime.now().date():
                        last_updated_count += 1
                    last_updated = file_time.strftime("%Y-%m-%d %H:%M")

                forecasts_with_prices[ticker] = {
                    "next_day_prediction": next_day_prediction,
                    "current_price": price,
                    "change_percentage": change_percentage,
                    "last_updated": last_updated,
                }
            except Exception as inner_e:
                print(f"Error processing {ticker}: {inner_e}")
                continue

        avg_prediction_change = (
            (sum(prediction_changes) / len(prediction_changes))
            if prediction_changes
            else 0
        )
        last_update = (
            forecaster.last_update.strftime("%Y-%m-%d %H:%M")
            if forecaster.last_update
            else None
        )

        return render_template(
            "SP500_Results.html",
            forecasts=forecasts_with_prices,
            total_forecasts=total_forecasts,
            last_updated_count=last_updated_count,
            avg_prediction_change=avg_prediction_change,
            last_update=last_update,
        )

    except Exception as e:
        print(f"Error in SP500 direct route: {e}")
        return render_template(
            "SP500_Results.html",
            error=f"Error loading forecasts: {str(e)}",
            forecasts={},
            total_forecasts=0,
            last_updated_count=0,
            avg_prediction_change=0,
            last_update=(
                forecaster.last_update.strftime("%Y-%m-%d %H:%M")
                if forecaster.last_update
                else None
            ),
        )


from flask_cachecontrol import dont_cache


@app.route("/LSTM", methods=("GET", "POST"))
@dont_cache()
def LSTM():
    """Legacy LSTM route for individual stock prediction"""
    if request.method == "POST":
        raw_ticker = request.form["ticker"].strip().upper()
        # Sanitize (map '.' to '-' for tickers like BRK.B)
        ticker = raw_ticker.replace(".", "-")
        if not ticker:
            flash(message="Please enter a valid Stock ticker")
            return render_template("LSTM.html")

        # Redirect to streaming progress page
        return redirect(url_for("lstm_progress", ticker=ticker))

    return render_template("LSTM.html")


@app.route("/LSTM/progress/<ticker>")
@dont_cache()
def lstm_progress(ticker):
    """Progress page for LSTM processing"""
    return render_template("LSTM_Progress.html", ticker=ticker)


# Global storage for LSTM results and status
lstm_results_cache = {}
lstm_status = {}

# Add a lock for thread safety when accessing shared resources
import threading

lstm_lock = threading.Lock()


@app.route("/LSTM/stream/<ticker>")
@dont_cache()
def lstm_stream(ticker):
    """Streaming endpoint for LSTM processing"""

    # Check if this ticker is already being processed
    if ticker in active_lstm_sessions:

        def already_processing():
            yield 'data: {"step": "error", "message": "LSTM processing for this ticker is already in progress. Please wait for it to complete."}\n\n'

        return Response(already_processing(), mimetype="text/plain")

    # Mark this ticker as being processed
    active_lstm_sessions[ticker] = True
    lstm_status[ticker] = {"step": 0, "message": "Starting...", "progress": 0}

    def generate():
        try:
            import time as time_module

            # Step 1: Download data
            lstm_status[ticker] = {
                "step": 1,
                "message": "Downloading stock data...",
                "progress": 10,
            }
            yield 'data: {"step": 1, "message": "Downloading stock data...", "progress": 10}\n\n'

            import pandas as pd
            import yfinance as yf
            import numpy as np
            from sklearn.preprocessing import MinMaxScaler
            import tensorflow as tf
            from keras import optimizers
            from keras.callbacks import History
            from keras.models import Model
            from keras.models import Sequential
            from keras.layers import (
                LSTM,
                Dense,
                Dropout,
                TimeDistributed,
                Input,
                Activation,
                Concatenate,
            )
            import matplotlib

            matplotlib.use("agg")
            from matplotlib import pyplot as plt
            import seaborn as sns

            lstm_status[ticker] = {
                "step": 2,
                "message": "Downloading stock data...",
                "progress": 20,
            }
            yield 'data: {"step": 2, "message": "Downloading stock data...", "progress": 20}\n\n'

            df = yf.download(tickers=ticker, period="3y")
            if df is None or df.empty:
                lstm_status[ticker] = {
                    "step": "error",
                    "message": "No data returned for ticker. Please check the symbol or try later.",
                }
                yield 'data: {"step": "error", "message": "No data returned for ticker. Please check the symbol or try later."}\n\n'
                return

            lstm_status[ticker] = {
                "step": 3,
                "message": "Preprocessing data...",
                "progress": 30,
            }
            yield 'data: {"step": 3, "message": "Preprocessing data...", "progress": 30}\n\n'

            og_dates = df.index
            df["EMAF"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMAM"] = df["Close"].ewm(span=60, adjust=False).mean()
            df["EMAS"] = df["Close"].ewm(span=100, adjust=False).mean()

            def compute_rsi(series, period=15):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df["RSI"] = compute_rsi(df["Close"], period=15)
            df["Target"] = df["Close"].shift(-1)
            df = df.dropna()
            if df.empty or len(df) < 60:
                lstm_status[ticker] = {
                    "step": "error",
                    "message": "Insufficient historical data after preprocessing.",
                }
                yield 'data: {"step": "error", "message": "Insufficient historical data after preprocessing."}\n\n'
                return

            lstm_status[ticker] = {
                "step": 4,
                "message": "Preparing model data...",
                "progress": 40,
            }
            yield 'data: {"step": 4, "message": "Preparing model data...", "progress": 40}\n\n'

            dates = df.index
            df.index = range(len(dates))
            df["Date"] = dates
            og_df = df
            cols = list(df)
            cols.pop(3)
            cols.pop(4)
            cols.pop(-1)
            df = df[cols].astype(float)
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled = scaler.fit_transform(df)
            X_train = []
            pastcandles = 30
            num_features = df.shape[1]
            for i in range(num_features):
                X_train.append([])
                for j in range(pastcandles, df_scaled.shape[0]):
                    X_train[i].append(df_scaled[j - pastcandles : j, i])
            X_train = np.moveaxis(X_train, [0], [2])
            X_train = np.array(X_train)
            Y = np.array(df_scaled[pastcandles:, -1])
            Y = np.reshape(Y, (len(Y), 1))
            ratio = int(len(X_train) * 0.8)
            if ratio == 0 or len(X_train) - ratio == 0:
                lstm_status[ticker] = {
                    "step": "error",
                    "message": "Not enough data to split into train/test sets.",
                }
                yield 'data: {"step": "error", "message": "Not enough data to split into train/test sets."}\n\n'
                return
            X_train, X_test = X_train[:ratio], X_train[ratio:]
            Y_train, Y_test = Y[:ratio], Y[ratio:]

            lstm_status[ticker] = {
                "step": 5,
                "message": "Building LSTM model...",
                "progress": 50,
            }
            yield 'data: {"step": 5, "message": "Building LSTM model...", "progress": 50}\n\n'

            lstm = Input(shape=(pastcandles, num_features), name="LSTM_input")
            lstm_input = LSTM(150, name="First_layer")(lstm)
            lstm_input = Dense(1, name="Dense_layer")(lstm_input)
            result = Activation("linear", name="Result")(lstm_input)
            model = Model(inputs=lstm, outputs=result)
            adam = optimizers.Adam()
            model.compile(optimizer=adam, loss="mse")

            lstm_status[ticker] = {
                "step": 6,
                "message": "Training LSTM model (this may take 2-3 minutes)...",
                "progress": 60,
            }
            yield 'data: {"step": 6, "message": "Training LSTM model (this may take 2-3 minutes)...", "progress": 60}\n\n'

            model.fit(
                x=X_train,
                y=Y_train,
                batch_size=15,
                epochs=30,
                shuffle=True,
                validation_split=0.1,
            )

            lstm_status[ticker] = {
                "step": 7,
                "message": "Generating predictions...",
                "progress": 70,
            }
            yield 'data: {"step": 7, "message": "Generating predictions...", "progress": 70}\n\n'

            Prediction = model.predict(X_test)
            Y_pred = scaler.inverse_transform(
                np.repeat(Prediction, df_scaled.shape[1], axis=-1)
            )[:, 0]
            from pandas.tseries.holiday import USFederalHolidayCalendar
            from pandas.tseries.offsets import CustomBusinessDay

            us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            forecast_dates = pd.date_range(
                list(dates)[-len(X_test)], periods=len(Y_pred), freq=us_bd
            ).tolist()
            prediction_dates = [date.date() for date in forecast_dates]
            forecast = pd.DataFrame(
                {"Date": np.array(prediction_dates), "Target": Y_pred}
            )
            forecast["Date"] = pd.to_datetime(forecast["Date"])
            lst_cols = ["Target", "Date"]
            original_df = og_df[lst_cols]
            original_df["Date"] = pd.to_datetime(original_df["Date"])
            original_df = original_df.loc[original_df["Date"] >= "2022-01-01"]

            lstm_status[ticker] = {
                "step": 8,
                "message": "Creating visualizations...",
                "progress": 80,
            }
            yield 'data: {"step": 8, "message": "Creating visualizations...", "progress": 80}\n\n'

            import matplotlib.pyplot as plt
            import seaborn as sns

            figure = plt.figure(figsize=(10, 5))
            sns.lineplot(
                x=original_df["Date"],
                y=original_df["Target"],
                label="Original closing prices",
            )
            sns.lineplot(
                x=forecast["Date"],
                y=forecast["Target"],
                label="Forecasted closing prices",
            )
            plt.legend()
            os.makedirs("static/Image", exist_ok=True)
            # Make image paths unique per ticker to avoid conflicts
            verification_path = os.path.join(
                "static/Image", f"verification_{ticker}.png"
            )
            figure.savefig(verification_path)
            futurecandles = 10
            futuredates = pd.date_range(
                list(og_dates)[-1], periods=futurecandles, freq=us_bd
            ).tolist()
            futuredates = pd.to_datetime(futuredates)
            future_prediction = model.predict(X_test[-futurecandles:])
            future_prediction = scaler.inverse_transform(
                np.repeat(future_prediction, df_scaled.shape[1], axis=-1)
            )[:, 0]
            figure1 = plt.figure(figsize=(10, 5))
            sns.lineplot(
                x=futuredates,
                y=future_prediction,
                label="Future closing prices prediction",
            )
            plt.legend()
            prediction_path = os.path.join("static/Image", f"prediction_{ticker}.png")
            figure1.savefig(prediction_path)

            # Close figures to free memory
            plt.close(figure)
            plt.close(figure1)

            lstm_status[ticker] = {
                "step": 9,
                "message": "Finalizing results...",
                "progress": 90,
            }
            yield 'data: {"step": 9, "message": "Finalizing results...", "progress": 90}\n\n'

            # Store results for later access
            lstm_results_cache[ticker] = {
                "status": "completed",
                "timestamp": time_module.time(),
            }

            # Send completion message
            lstm_status[ticker] = {
                "step": "complete",
                "message": "Processing complete! Redirecting to results...",
                "progress": 100,
                "redirect": f"/LSTM/results/{ticker}",
            }
            yield f'data: {{"step": "complete", "message": "Processing complete! Redirecting to results...", "progress": 100, "redirect": "/LSTM/results/{ticker}"}}\n\n'

        except Exception as e:
            import traceback

            print("Exception in LSTM progress:", e)
            traceback.print_exc()
            lstm_status[ticker] = {"step": "error", "message": f"Error: {str(e)}"}
            yield f'data: {{"step": "error", "message": "Error: {str(e)}"}}\n\n'
        finally:
            # Always clean up the session when done
            if ticker in active_lstm_sessions:
                del active_lstm_sessions[ticker]

            # Clean up old image files (keep only the most recent ones)
            try:
                import glob

                # Keep only the last 10 image files per type to prevent disk space issues
                for pattern in [
                    f"static/Image/verification_*.png",
                    f"static/Image/prediction_*.png",
                ]:
                    files = glob.glob(pattern)
                    if len(files) > 10:
                        # Sort by modification time and remove oldest
                        files.sort(key=lambda x: os.path.getmtime(x))
                        for old_file in files[:-10]:
                            try:
                                os.remove(old_file)
                            except:
                                pass  # Ignore errors if file is already deleted
            except Exception as e:
                print(f"Error cleaning up old image files: {e}")

    return Response(generate(), mimetype="text/plain")


@app.route("/LSTM/results/<ticker>")
@dont_cache()
def lstm_results(ticker):
    """Display LSTM results"""
    try:
        # Check if the ticker-specific image files exist
        verification_path = os.path.join("static/Image", f"verification_{ticker}.png")
        prediction_path = os.path.join("static/Image", f"prediction_{ticker}.png")

        if not os.path.exists(verification_path) or not os.path.exists(prediction_path):
            flash("Results not found. Please try running the prediction again.")
            return redirect(url_for("LSTM"))

        return render_template(
            "LSTM_Results.html",
            img1=f"/static/Image/verification_{ticker}.png",
            img2=f"/static/Image/prediction_{ticker}.png",
            ticker=ticker,
        )
    except Exception as e:
        flash(f"Error displaying results: {str(e)}")
        return redirect(url_for("LSTM"))


# Legacy LSTM route (kept for backward compatibility)
@app.route("/LSTM/legacy", methods=("GET", "POST"))
@dont_cache()
def LSTM_legacy():
    """Legacy LSTM route for individual stock prediction"""
    if request.method == "POST":
        raw_ticker = request.form["ticker"].strip().upper()
        # Sanitize (map '.' to '-' for tickers like BRK.B)
        ticker = raw_ticker.replace(".", "-")
        if not ticker:
            flash(message="Please enter a valid Stock ticker")
            return render_template("LSTM.html")
        try:
            import pandas as pd
            import yfinance as yf
            import numpy as np
            from sklearn.preprocessing import MinMaxScaler
            import tensorflow as tf
            from keras import optimizers
            from keras.callbacks import History
            from keras.models import Model
            from keras.models import Sequential
            from keras.layers import (
                LSTM,
                Dense,
                Dropout,
                TimeDistributed,
                Input,
                Activation,
                Concatenate,
            )
            import matplotlib

            matplotlib.use("agg")
            from matplotlib import pyplot as plt
            import seaborn as sns

            df = yf.download(tickers=ticker, period="3y")
            if df is None or df.empty:
                flash(
                    "No data returned for ticker. Please check the symbol or try later."
                )
                return render_template("LSTM.html")

            og_dates = df.index
            df["EMAF"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMAM"] = df["Close"].ewm(span=60, adjust=False).mean()
            df["EMAS"] = df["Close"].ewm(span=100, adjust=False).mean()

            def compute_rsi(series, period=15):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df["RSI"] = compute_rsi(df["Close"], period=15)
            df["Target"] = df["Close"].shift(-1)
            df = df.dropna()
            if df.empty or len(df) < 60:
                flash("Insufficient historical data after preprocessing.")
                return render_template("LSTM.html")
            dates = df.index
            df.index = range(len(dates))
            df["Date"] = dates
            og_df = df
            cols = list(df)
            cols.pop(3)
            cols.pop(4)
            cols.pop(-1)
            df = df[cols].astype(float)
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled = scaler.fit_transform(df)
            X_train = []
            pastcandles = 30
            num_features = df.shape[1]
            for i in range(num_features):
                X_train.append([])
                for j in range(pastcandles, df_scaled.shape[0]):
                    X_train[i].append(df_scaled[j - pastcandles : j, i])
            X_train = np.moveaxis(X_train, [0], [2])
            X_train = np.array(X_train)
            Y = np.array(df_scaled[pastcandles:, -1])
            Y = np.reshape(Y, (len(Y), 1))
            ratio = int(len(X_train) * 0.8)
            if ratio == 0 or len(X_train) - ratio == 0:
                flash("Not enough data to split into train/test sets.")
                return render_template("LSTM.html")
            X_train, X_test = X_train[:ratio], X_train[ratio:]
            Y_train, Y_test = Y[:ratio], Y[ratio:]
            lstm = Input(shape=(pastcandles, num_features), name="LSTM_input")
            lstm_input = LSTM(150, name="First_layer")(lstm)
            lstm_input = Dense(1, name="Dense_layer")(lstm_input)
            result = Activation("linear", name="Result")(lstm_input)
            model = Model(inputs=lstm, outputs=result)
            adam = optimizers.Adam()
            model.compile(optimizer=adam, loss="mse")
            model.fit(
                x=X_train,
                y=Y_train,
                batch_size=15,
                epochs=30,
                shuffle=True,
                validation_split=0.1,
            )
            Prediction = model.predict(X_test)
            Y_pred = scaler.inverse_transform(
                np.repeat(Prediction, df_scaled.shape[1], axis=-1)
            )[:, 0]
            from pandas.tseries.holiday import USFederalHolidayCalendar
            from pandas.tseries.offsets import CustomBusinessDay

            us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            forecast_dates = pd.date_range(
                list(dates)[-len(X_test)], periods=len(Y_pred), freq=us_bd
            ).tolist()
            prediction_dates = [date.date() for date in forecast_dates]
            forecast = pd.DataFrame(
                {"Date": np.array(prediction_dates), "Target": Y_pred}
            )
            forecast["Date"] = pd.to_datetime(forecast["Date"])
            lst_cols = ["Target", "Date"]
            original_df = og_df[lst_cols]
            original_df["Date"] = pd.to_datetime(original_df["Date"])
            original_df = original_df.loc[original_df["Date"] >= "2022-01-01"]
            import matplotlib.pyplot as plt
            import seaborn as sns

            figure = plt.figure(figsize=(10, 5))
            sns.lineplot(
                x=original_df["Date"],
                y=original_df["Target"],
                label="Original closing prices",
            )
            sns.lineplot(
                x=forecast["Date"],
                y=forecast["Target"],
                label="Forecasted closing prices",
            )
            plt.legend()
            os.makedirs("static/Image", exist_ok=True)
            # Make image paths unique per ticker to avoid conflicts
            verification_path = os.path.join(
                "static/Image", f"verification_{ticker}.png"
            )
            figure.savefig(verification_path)
            futurecandles = 10
            futuredates = pd.date_range(
                list(og_dates)[-1], periods=futurecandles, freq=us_bd
            ).tolist()
            futuredates = pd.to_datetime(futuredates)
            future_prediction = model.predict(X_test[-futurecandles:])
            future_prediction = scaler.inverse_transform(
                np.repeat(future_prediction, df_scaled.shape[1], axis=-1)
            )[:, 0]
            figure1 = plt.figure(figsize=(10, 5))
            sns.lineplot(
                x=futuredates,
                y=future_prediction,
                label="Future closing prices prediction",
            )
            plt.legend()
            prediction_path = os.path.join("static/Image", f"prediction_{ticker}.png")
            figure1.savefig(prediction_path)

            # Close figures to free memory
            plt.close(figure)
            plt.close(figure1)

            return render_template(
                "LSTM_Results.html",
                img1=f"/static/Image/verification_{ticker}.png",
                img2=f"/static/Image/prediction_{ticker}.png",
                ticker=raw_ticker,
            )
        except Exception as e:
            import traceback

            print("Exception in /LSTM:", e)
            traceback.print_exc()
            flash("Internal error while generating forecast. Please try again later.")
    return render_template("LSTM.html")


# --- Optional background updater (explicit; no work at import time) ---
try:
    from sp500_forecaster import start_background_updates

    if os.environ.get("ENABLE_BACKGROUND_UPDATES", "0") == "1":
        start_background_updates()
except Exception as e:
    # Do not block app startup if the background thread fails
    try:
        app.logger.exception("Failed to start background updates: %s", e)
    except Exception:
        print(f"Failed to start background updates: {e}")

if __name__ == "__main__":
    app.run(debug=False)

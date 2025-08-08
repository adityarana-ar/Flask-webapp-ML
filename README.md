# ML Summer Project Web App

This repository contains a Flask web application for:

- Predicting diabetes using SVM (Support Vector Machine) from user input
- **S&P 500 Stock Forecasting System** - Automated daily forecasts for selected S&P 500 stocks
- Legacy LSTM-based time series forecasting for individual stock tickers

## Features

- **S&P 500 Forecasting**: Automated daily forecasts (cached on disk; UI never blocks generation)
- Real-time stock price comparison with predictions
- Background processing that updates forecasts daily after market close (4 PM US/Eastern)
- Web forms for diabetes prediction and individual stock price forecasting
- LSTM-based time series forecasting for any stock ticker (e.g., AAPL)
- Visualization of original and predicted stock prices

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/adityarana-ar/Flask-webapp-ML.git
cd Flask-webapp-ML
```

2. **Create and activate a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **(Optional: Initial Bulk Forecast Generation)**  
   If you want forecasts available immediately on first load of `/SP500`:

```bash
python generate_forecasts.py
```

Otherwise the background thread/service will begin generating after first start (can take a long time).

5. **Run the Flask app (development)**

```bash
python main.py
```

6. **Open your browser**

Go to http://127.0.0.1:5000

## Usage

### /SP500 (Batch Forecast Dashboard)

- Visit `/SP500` to view cached forecasts (next-day predicted close) and current market prices.
- The page only reads existing CSV forecast files from `forecast_data/predictions/` and does NOT trigger heavy model retraining.
- If you see a message saying no cached forecasts, either run `python generate_forecasts.py` or wait for the background update to complete.

### /LSTM (Individual Stock Forecast)

- Enter a ticker (e.g., AAPL). Symbols with dots use a dash (e.g., BRK.B -> BRK-B) automatically.
- Produces historical fit visualization + 10 future business-day predictions.

### /Diabetes

- Fill out the form; result returns diabetic / non-diabetic classification.

## Background Forecast Updates

A lightweight scheduler runs inside the Flask process (background thread) and checks hourly whether forecasts should be regenerated (once per trading day after market close).

For production you should externalize this using one of:

1. Cron + standalone script:
   ```cron
   # Run at 22:00 UTC (after US market close) daily Mon-Fri
   0 22 * * 1-5 /path/to/venv/bin/python /path/to/app/generate_forecasts.py >> /var/log/sp500_forecasts.log 2>&1
   ```
2. systemd service + timer (preferred long‑running model if you want continuous web + background):
   - Provided example service: `sp500-forecaster.service`
   - Create a complementary timer `sp500-forecaster.timer` OR keep service always-on and let internal thread handle scheduling.

### Example systemd timer (create `/etc/systemd/system/sp500-forecaster.timer`):

```ini
[Unit]
Description=Daily S&P 500 Forecast Generation Timer

[Timer]
OnCalendar=Mon..Fri 22:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:

```bash
sudo systemctl enable sp500-forecaster.service
sudo systemctl enable sp500-forecaster.timer
sudo systemctl start sp500-forecaster.service
sudo systemctl start sp500-forecaster.timer
```

## Data & Artifacts

- Models: `forecast_data/models/*.h5`
- Scalers: `forecast_data/scalers/*.pkl`
- Forecast CSVs: `forecast_data/predictions/*_forecast.csv`
- Summary JSON: `forecast_data/forecast_summary.json`
- Last update timestamp: `forecast_data/last_update.json`
- Plots (individual LSTM page): `static/Image/`

## Deployment Notes

- Start small (single Ubuntu LTS droplet / EC2) with: gunicorn + nginx reverse proxy.
- For cost efficiency: generate forecasts off-peak (after close) and serve static CSVs during day.
- Scale path: add task queue (Celery / RQ) + Redis if concurrency or queueing needed.
- Avoid retraining during user requests; keep read-only endpoints fast.

### Gunicorn Example

```bash
gunicorn -w 2 -b 0.0.0.0:8000 main:app
```

### Nginx Snippet

```
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## Docker (Optional)

Create a `Dockerfile` (not yet included) for reproducible builds if containerizing. Example base:

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "main:app"]
```

## Testing Ideas (Not Implemented)

- Unit test `SP500Forecaster.prepare_data` for empty DataFrame handling.
- Test `get_all_forecasts()` returns only existing CSVs.
- Mock yfinance to ensure retry / failure branches work.

## Future Improvements

- Add retry/backoff wrapper around yfinance downloads.
- Implement caching layer (e.g., shelve or simple JSON TTL) for intraday price fetches.
- Introduce async task queue if expanding ticker coverage.
- Add API endpoints (JSON) for programmatic access.

## License

Educational use only.

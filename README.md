# ML Summer Project Web App

This repository contains a Flask web application for:

- Predicting diabetes using SVM (Support Vector Machine) from user input
- Forecasting stock prices using an LSTM neural network

## Features

- Web forms for diabetes prediction and stock price forecasting
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

4. **Run the Flask app**

```bash
python main.py
```

5. **Open your browser**

Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage

- For diabetes prediction, fill out the form on the `/Diabetes` page.
- For stock price forecasting, enter a valid stock ticker (e.g., `AAPL`) on the `/LSTM` page.
- Results and plots will be displayed after form submission.

## Notes

- Plots are saved in `static/Image/` and displayed on the results pages.
- Make sure you have an active internet connection for yfinance data download.

## License

This project is for educational purposes.

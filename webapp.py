import os
from flask import Flask, request, redirect, url_for, render_template

app = Flask(__name__)

# Importing dependencies
import pandas as pd
import yfinance as yf
import numpy as np
# ! pip install pandas_ta==0.2.45b
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Activation
import seaborn as sns
from pandas.core.api import DateOffset

# Defining function that makes the model
# and the prediction

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        ticker = request.form.get("ticker")
        pastcandles = request.form.get("past_candles")
        futurecandles = request.form.get("future_candles")
        comparison_date = request.form.get("date")
        return redirect(url_for('prediction',ticker=ticker, pastcandles=pastcandles, date=comparison_date))
    return render_template('index.html')

@app.route('/prediction')
def prediction(ticker: str, pastcandles: int, futurecandles: int, date: str):
    # Importing dataframe from yahoo finance
    df = yf.download(tickers = ticker)
    # print(df) # preview of df

    # Adding other indicators

    og_dates = df.index
    df['RSI'] = ta.rsi(df.Close, length=15) #RSI
    df['EMAF'] = ta.ema(df.Close, length=20) # Exponential Moving Average with short period
    df['EMAM'] = ta.ema(df.Close, length=60) # EMA with medium period
    df['EMAS'] = ta.ema(df.Close, length=100) # EMA with long period
    df['Target'] = df['Adj Close'].shift(-1) # Next day's Adjusted Closing price
    df = df.dropna()
    # Separating the dates as they are the index here
    dates = df.index
    indeces = []
    for i in range(len(dates)):
        indeces.append(i)
    df.index = indeces
    df['Date'] = dates
    # print(df.index)
    # df.head(20)
    og_df = df
    # print(dates)
    # print(dates)
    # print(og_df)

    cols = list(df)
    # print(cols)

    cols.pop(3)
    cols.pop(4)
    cols.pop(-1)
    # print(cols)

    df = df[cols].astype(float)

    # df.head()

    # print(df)

    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df)
    # print(df_scaled)
    # print(df_scaled.shape)

    X_train = []

    # df_scaled.shape

    # pastcandles = 10 # past number of days to train the model on

    for i in range(9): # appending columns and values for the pastcandles to X_train
        X_train.append([])
    for j in range(pastcandles, df_scaled.shape[0]):
        X_train[i].append(df_scaled[j-pastcandles:j, i])

    # print(X_train)
    # Moving X axis from 0 to 2
    X_train = np.moveaxis(X_train, [0], [2])

    X_train = np.array(X_train)
    Y = np.array(df_scaled[pastcandles:,-1])
    # Reshaping Y
    Y = np.reshape(Y, (len(Y), 1))

    # print("Shape of X: ", X_train.shape)
    # print("Shape of Y: ", Y.shape)

    # Splitting between training and testing data

    ratio = int(len(X_train)*0.8)
    X_train, X_test = X_train[:ratio], X_train[ratio:]
    Y_train, Y_test = Y[:ratio], Y[ratio:]

    # Verifying shapes
    # print("X_train shape: ", X_train.shape)
    # print("X_test shape: ", X_test.shape)
    # print("Y_train shape: ", Y_train.shape)
    # print("Y_test shape: ", Y_test.shape)

    # Making the model

    lstm = Input(shape=(pastcandles, 9), name='LSTM_input')
    lstm_input = LSTM(150, name='First_layer')(lstm) # LSTM layer with 150 nodes
    lstm_input = Dense(1, name='Dense_layer')(lstm_input) # Dense layer with 1 node
    result = Activation('linear', name='Result')(lstm_input)
    model = Model(inputs=lstm, outputs=result)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=Y_train, batch_size=15, epochs=30, shuffle=True, validation_split=0.1)

    # lstm_input = Input(shape=(pastcandles, 9), name='lstm_input')
    # inputs = LSTM(150, name='first_layer')(lstm_input)
    # inputs = Dense(1, name='dense_layer')(inputs)
    # output = Activation('linear', name='output')(inputs)
    # model = Model(inputs=lstm_input, outputs=output)
    # adam = optimizers.Adam()
    # model.compile(optimizer=adam, loss='mse')
    # model.fit(x=X_train, y=Y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)

    Prediction = model.predict(X_test)

    # for i in range(10):
    #   print(Prediction[i], Y_test[i])

    # Using inverse transform to turn it into stock prices for comparison

    Y_pred = scaler.inverse_transform(np.repeat(Prediction, df_scaled.shape[1], axis=-1))[:,0]

    # Adding dates to the predicition

    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    forecast_dates = pd.date_range(list(dates)[-len(X_test)], periods=len(Y_pred), freq=us_bd).tolist()

    prediction_dates = []

    for time in forecast_dates:
        prediction_dates.append(time.date())

    # print(prediction_dates)

    forecast = pd.DataFrame({'Date':np.array(prediction_dates), 'Target':Y_pred})
    forecast['Date'] = pd.to_datetime(forecast['Date'])
    # original_df = df['Target']

    # og_dates = dates>='2022-08-22'

    lst = ['Target', 'Date']


    original_df = og_df[lst]
    original_df['Date'] = pd.to_datetime(original_df['Date'])
    original_df = original_df.loc[original_df['Date']>=date]

    # list of plots to return
    result = []

    figure1 = sns.lineplot(x=original_df['Date'], y=original_df['Target'])
    figure1 = sns.lineplot(x=forecast['Date'], y=forecast['Target'])

    # figure1 = plt.show(False)
    result.append(figure1)

    futuredates = pd.date_range(list(og_dates)[-1], periods=futurecandles, freq=us_bd).tolist()

    # for day in futuredates:
    #   print(day + DateOffset(day=2))
    # print(type(futuredates))
    futuredates = pd.to_datetime(futuredates)
    # print(futuredates)
    # print(dates)

    future_prediction = model.predict(X_test[-futurecandles:])
    # future_prediction = np.repeat(future_prediction, df_scaled.shape[1], axis=-1)
    future_prediction = scaler.inverse_transform(np.repeat(future_prediction, df_scaled.shape[1], axis=-1))[:,0]
    # print(Y_pred.shape)

    output = pd.DataFrame({'Date':np.array(futuredates)})
    output['Target'] = future_prediction


    # Plotting graph 

    figure2 = sns.lineplot(x=futuredates, y=future_prediction)

    result.append(figure2)
    # plt.show()
    # figure = graph.get_figure()
    # figure.savefig("output.png")

    return result


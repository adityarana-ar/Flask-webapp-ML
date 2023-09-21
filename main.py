import os
from flask import Flask, render_template, request, url_for, flash, redirect
# from flask_caching import Cache
from flask_cachecontrol import (
    dont_cache,
)

app = Flask(__name__)
img = os.path.join('static', 'Image')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = os.urandom(24).hex()

@app.route("/")
def home():
    return render_template("index.html")

d_input = [{'pregnancies': 0,
            'glucose': 0,
            'bp': 0,
            'skin_thickness': 0,
            'insulin': 0,
            'bmi': 0,
            'dpf': 0,
            'age': 0}]



@app.route("/Diabetes", methods=('GET', 'POST'))
def diabetes():
    if request.method == 'POST':
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['blood_pressure']
        skin_thickness = request.form['skin_thickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

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
            d_input.append({'pregnancies': pregnancies, 'glucose': glucose,
                            'bp': bp, 'skin_thickness': skin_thickness,
                            'insulin': insulin, 'bmi': bmi,
                            'dpf': dpf, 'age': age})
            
            import numpy as np # To make numpy arrays
            import pandas as pd # Used to create data frames (tables)
            from sklearn.preprocessing import StandardScaler # To standardize the data
            from sklearn.model_selection import train_test_split # To split the data into
            # training and testing

            from sklearn import svm # Importing Support Vector Machine model
            from sklearn.metrics import accuracy_score

            """Data Collection and Analysis"""

            # We can probably change the data or switch to a different kind of data.
            # this is from a tutorial just to see how well the model works.

            # Loading diabetes.csv through panda dataframe

            df = pd.read_csv('diabetes.csv')

            # Printing first 5 rows of dataset
            # df.head()

            # df.describe()

            # df['Pregnancies'].value_counts()

            # df['Outcome'].value_counts()

            # Separating data and labels
            X = df.drop(columns='Outcome', axis=1) # column of outcome is dropped. rest is assigned
            Y = df['Outcome'] # Only outcome is stored corresponding to df's indexes

            # print(X)

            # Standardizing data now since all values have differenct ranges.
            scaler = StandardScaler()

            standardized_data = scaler.fit_transform(X) # Fitting and transforming X into
            # variable standardized_data
            # print(standardized_data)

            # Ranges are similar across different columns
            X = standardized_data
            Y = df['Outcome']

            # print(Y)

            # Splitting data into Training data and test data
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
            # 4 variables here, X being split in train and test
            # 0.2 = 20% of data (how much of data is test data)
            # Stratify is to ensure that the X_train, X_test have the same proportion (diabestic, non-diabetic) as data
            # So it will eliminate all non-diabetic data going to train and it being tested on all diabetic people
            # Random state is to replicate the same splitting for testing purposes ig

            # print(X.shape, X_train.shape, X_test.shape)

            # 20% went to X_test rest 80% is with X_train (i.e. more date to train which is good)

            # TRAINING THE MODEL
            # Loading support vector machine
            classifier = svm.SVC(kernel='linear')

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

            input_data = (pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age)

            # Change it to numpy array

            input_data = np.asarray(input_data)

            # Re-shaping the array (as model is expecting 768 values rn)

            input_data = input_data.reshape(1,-1)

            # Standardizing the data

            input_data = scaler.transform(input_data)

            # print(input_data)

            prediction = classifier.predict(input_data)
            # print(prediction)
            if (prediction == 0):
                result = "non-diabetic"
            else:
                result = "diabetic"

            return render_template("Diabetes_Results.html", prediction=result)
        except:
            flash("An error ocurred. Try using valid parameters for the web form.")

    return render_template("Diabetes.html")
ticker = ""
@app.route("/LSTM", methods=('GET', 'POST'))
@dont_cache()
def LSTM():
    if request.method == 'POST':
        ticker = request.form['ticker']
        # pcandles = request.form['pcandles']
        # fcandles = request.form['fcandles']
        if not ticker:
            flash(message="Please enter a valid Stock ticker")
        # if not pcandles:
        #     flash(message="You must enter a valid number for the number of past candles to check.")
        # if not fcandles:
        #     flash(message="You must enter a valid number for the number of days to predict the closing prices for.")
        try:
            import pandas as pd
            import yfinance as yf
            import numpy as np
            # ! pip install pandas_ta==0.2.45b
            import pandas_ta as ta
            from sklearn.preprocessing import MinMaxScaler
            import tensorflow as tf
            from keras import optimizers
            from keras.callbacks import History
            from keras.models import Model
            from keras.models import Sequential
            from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Input, Activation, Concatenate
            import matplotlib
            matplotlib.use('agg')
            from matplotlib import pyplot as plt
            import seaborn as sns
            # Importing dataframe from yahoo finance
            df = yf.download(tickers = ticker, period="3y")

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

            df_scaled.shape

            pastcandles = 30 # past number of days to train the model on

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
            original_df = original_df.loc[original_df['Date']>='2022-01-01']

            # original_df = pd.DateFrame({'Date':dates>='2022'})
            # print(original_df.head(20))


            # indeces = []
            # for i in range(len(original_df)):
            #   indeces.append(i)
            # original_df.index= indeces
            # original_df['Date'] = pd.to_datetime(original_df['Date'])
            # original_df = original_df.loc[original_df['Date']>='2022-8-22']

            # # Plotting
            figure = plt.figure(figsize=(10,5))

            sns.lineplot(x=original_df['Date'], y=original_df['Target'], label="Original closing prices")
            sns.lineplot(x=forecast['Date'], y=forecast['Target'], label="Forecasted closing prices")
            plt.legend()
            figure.savefig("static/Image/verification.png")
            # plt.show()
            # Now predicting for the next 5 days

            futurecandles = 10
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
            figure1 = plt.figure(figsize=(10,5))

            sns.lineplot(x=futuredates, y=future_prediction, label="Future closing prices prediction")
            plt.legend()
            figure1.savefig("static/Image/prediction.png")
            # plt.show()

            # Importing image to be read
            from PIL import Image
            import base64
            import io
            
            file1 = os.path.join(img, 'verification.png')
            # os.remove("verification.png")
            file2 = os.path.join(img, 'prediction.png')
            # os.remove("prediction.png")


            # , img1= img1, img2= img2
            return render_template("LSTM_Results.html", img1= file1, img2= file2, ticker=ticker)
        except:
            flash("An error ocurred. Try using valid parameters for the web form.")
    return render_template("LSTM.html")

if __name__ == "__main__":
    app.run(debug=True)
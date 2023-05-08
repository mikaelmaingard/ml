# ML classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# data manipulation
import pandas as pd
import numpy as np
# plotting
import matplotlib.pyplot as plt
import seaborn
# data
import yfinance as yf

if __name__ == "__main__":
    df = yf.download('SPY', start="2017-01-01", end="2022-01-01")
    df = df.dropna()
    print("DF:\n", df)
    df.Close.plot(figsize=(10,5))
    plt.ylabel("S&P500 Price")
    plt.show()

    '''
      Target variable is whether or not close price will be 
      up on next trading day. 1 for up, -1 for down.
    '''
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    print("Y:\n", y)

    '''
        Features or predictor variables are used to predict
        target variable. We will use OHLC (Open-High-Low-Close).
    '''
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    X = df[['Open-Close', 'High-Low']]
    print("Pre-test dataframe:\n", df)

    '''
        Now we need to train the model. We use 80% of the data
        to train the model, and we test on the remaining 20%.
    '''
    training_split = 0.8
    split = int(training_split*len(df))
    # train dataset X,y
    X_train = X[:split]
    y_train = y[:split]
    # test dataset
    X_test = X[split:]
    y_test = y[split:]

    # create the ml classification model using the train dataset
    cls = SVC().fit(X_train, y_train)

    # check the accuracy of the training and testing phases
    accuracy_train = accuracy_score(y_train, cls.predict(X_train))
    accuracy_test = accuracy_score(y_test, cls.predict(X_test))
    print('\nTrain accuracy:{: .2f}'.format(accuracy_train*100))
    print('\nTest accuracy:{: .2f}'.format(accuracy_test*100))

    '''
        If we have a test accuracy > 50%, we have an effective model. Now 
        we can use the model to make predictions on the S&P500 close price.
    '''
    df['Predicted_Signal'] = cls.predict(X)
    # calculate log returns
    df['Return'] = np.log(df.Close.shift(-1) / df.Close)*100
    df['Strategy_Return'] = df.Return * df.Predicted_Signal
    # plot the cu,ulative strategy returns
    df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
    plt.ylabel("Strategy Returns (%)")
    plt.show()
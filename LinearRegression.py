# Using linear regression to predict future stock prices for certain companies
# Then generates a scatter plot graph with the linear regression line running
# through it.

import pandas as pd
import Quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from skylearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

df = Quandl.get("WIKI/GOOGL")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume",]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. CLose"] * 100.0
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

forecast_col = "Adj. Close"
df.fillna(-999999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)

df["label"] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(["label", "Adj. CLose"], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df["label"])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, text_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
with open("linearregression.pickle", "wb") as f:
    pickle.dump(clf, f)

# saves classifier so you do not have to retrain whenever you run this file
pickle_in = open("linearregression.pickle", "rb")
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy, forecast_out)
df["Forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# Creating predictive graph    
df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Data")
plt.xlabel("Price")
plt.show()

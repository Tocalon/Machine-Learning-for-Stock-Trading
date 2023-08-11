import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yfpip 

import chart_studio.plotly as py
import plotly.graph_objs as go 
from plotly.offline import plot, iplot

#for offline plotting 
#from plotly.offline import download_plotyjs, init_notebooke_mode, plot, iplot
#init_notebooke_mode(connected=True)

stock=pd.read_csv("/Users/joseacevedo/Desktop/Stock/AAPL_prices.csv")
stock.head()
stock.info()

stock["Date"]=pd.to_datetime(stock["Date"])
print('Dataframe contains sotck prices between {stock.Date.min()} {stock.Date.max()}')
print("Total days:  {stock.Date.max()}-{stock.Date.min()} days")

stock.describe()

stock[["Open", "High", "Low", "Close", "Adj Close"]].plot(kind="box")
layout=go.layout(
    tittle="Stock Prices", 
    xaxis=dict(
        title="Date", 
        titlefont=dict(
            family="Courier New, monospace", 
            size=18,
            color="#7f7f7f"
        )
    ), 
    yaxis=dict(
        title="Price",
        titlefont=dict(
            family="Courier New, monospace", 
            size=18,
            color="#7f7f7f"
        ) 
    )
)

stock_data=[{"x":stock["Date"], "y":stock["Close"]}]
plot=go.Figure(data=stock_data,layout=layout)

iplot(plot)

#buildign the regression model 
from sklearn.model_selection import train_test_split

#for prepocessing 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#for model evaluation 
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

#split the data into traind and test sets 
x=np.array(stock.index).reshape(-1,1)
y=stock["Close"]
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3, random_state=101)
scaler=StandardScaler().fit(x_train)
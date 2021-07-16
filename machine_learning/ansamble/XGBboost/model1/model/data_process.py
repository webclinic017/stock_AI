# import os
# import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# import joblib
# import operator
# visualize
# import matplotlib.pyplot as plt
# import matplotlib.style as style
# import seaborn as sns
# from matplotlib import pyplot
# from matplotlib.ticker import ScalarFormatter


def create_data(df):
    # simple moving average
    df['sma10'] = df['Adj Close'].rolling(30).mean()

    # Relative Strength Index in 5 days
    df['price-change-percentage'] = df['Adj Close'] / df['Adj Close'].shift(1)

    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    df['adosc-ema3'] = df['adosc'].ewm(span=3, adjust=False).mean()
    df['adosc-ema10'] = df['adosc'].ewm(span=10, adjust=False).mean()
    df['adosc-SG'] = np.where((df['adosc-ema3'] - df['adosc-ema10']) > 0, 1, -1)


    
    # set return and direction (label)
    df['return'] = 10*np.log(df['Adj Close'].shift(-5) / df['Adj Close'])

    AD = []
    AD.append(0)
    for i in range(1, len(df)):
        AD_component = ((df["Adj Close"][i] - df["Low"][i]) - (df["High"][i] - df["Adj Close"][i])) * df['Volume'][i] / (df["High"][i] - df["Low"][i]) + AD[-1]
        AD.append(AD_component)
    df['A/D'] = AD
    df['A/D_EMA'] = df['A/D'].ewm(com=20).mean()   
    df['A/D_ratio'] = df['A/D'] / df['A/D_EMA']
    df['ADOSC'] = df['A/D'].ewm(com=3).mean() / df['A/D'].ewm(com=10).mean()   
    df['A/D_EMA_ratio'] = df['A/D_EMA'] / df['A/D_EMA'].shift(1)
                
  

     # Commodity Channel Index in 24 days
    df['typical-price'] = (df['High'] + df['Low'] + df['Close']) / 3
   
    

    # simple moving average
    for i in [30,45,60,90,120,150,180,200,210,240,270,300]:
        df['sma'+str(i)] = df['Adj Close'].rolling(i).mean()
        df['sma'+str(i)] = (df['sma'+str(i)] / df['sma'+str(i)].shift(1)-1)*100
        df['today_by_sma'+str(i)+'ratio']= df['Adj Close']/df['sma'+str(i)] 
#         df['sma'+str(i)] = Zero_One_Scale(df['sma'+str(i)])
    for i in [10,30,60,90,120,150,180,200,210,240,270,300]:
        df["ema"+str(i)]=df["Adj Close"].ewm(span=i).mean()
        df["ema"+str(i)] = (df["ema"+str(i)] / df["ema"+str(i)].shift(1))*100
#         df["ema"+str(i)] = Zero_One_Scale(df["ema"+str(i)])
  

     #calucurate ADX
    df["TrueRange"] = np.nan
    df["PDM"] = np.nan
    df["NDM"] = np.nan
    for i in range(1,len(df)):
        df["TrueRange"][i] = TrueRange(df["Adj Close"][i],df["High"][i],df["Low"][i],df["Open"][i],df["Adj Close"][i-1])
        df["PDM"][i] = PDM(df["Open"][i],df["High"][i],df["Low"][i],df["Adj Close"][i],df["Open"][i-1],df["High"][i-1],df["Low"][i-1],df["Adj Close"][i-1])
        df["NDM"][i] = NDM(df["Open"][i],df["High"][i],df["Low"][i],df["Adj Close"][i],df["Open"][i-1],df["High"][i-1],df["Low"][i-1],df["Adj Close"][i-1])
    
    
    for i in range(15,150,30):
        for k in range(30,270,60):
            df['ratio_sma'+str(k)] = df['Adj Close'].rolling(k).mean()
            df['ratio_sma'+str(i)] = df['Adj Close'].rolling(i).mean()
            df['ratio_sma'+str(i)+'_'+str(k)] = (df['ratio_sma'+str(i)] / df['ratio_sma'+str(k)]-1)*10
#             df['ratio_sma'+str(i)+'_'+str(k)] = Zero_One_Scale(df['ratio_sma'+str(i)])

            
    for term in range(5,50,5):
        df['SMA'+str(term)] = df['Adj Close'].rolling(term).mean()
        df['STD'+str(term)] = df['Adj Close'].rolling(term).std()
        df['Standard_deviation_normalization'+str(term)] = np.log(100 * 2 * df['STD'+str(term)] / df['SMA'+str(term)])
#         df['Standard_deviation_normalization'+str(term)] = Zero_One_Scale(df['Standard_deviation_normalization'+str(term)])
        
    for i in [15,45,61,81,121,161]:
        df['Highest_in_range'+str(i)] = df['Adj Close'].rolling(window=i).max()
        df['Highest'+str(i)+'ago'] = (df['Highest_in_range'+str(i)].shift())
        for m in [30,81,90,150,60,60]:    
            df['Highest'+str(i)+','+str(m)+'days_ago'] = df['Adj Close'] / df['Highest'+str(i)+'ago'].shift(m)
#             df['Highest'+str(i)+','+str(m)+'days_ago'] = Zero_One_Scale(df['Highest'+str(i)+','+str(m)+'days_ago'])
        
        #今日の終値が過去何日間の高音に対してどの程度あるか
        df['Highest'+str(i)] = df['Adj Close'] / df['Highest_in_range'+str(i)]
#         df['Highest'+str(i)] = Zero_One_Scale(df['Highest'+str(i)])
        
    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    df['adosc_ratio'] = (df['adosc']/df['adosc'].shift(1)-1)*10

#     df['adosc'] = Zero_One_Scale(df['adosc'])

    # drop row contains NaN
    # df.dropna(inplace=True)
    return df

def TrueRange(c, h, l, o, yc):
    x = h-l
    y = abs(h-yc)
    z = abs(l-yc)
    if y <= x >= z:
        TR = x
    elif x <= y >= z:
        TR = y
    elif x <= z >= y:
        TR = z
    return TR

def PDM(o, h, l, c, yo, yh, yl, yc):
    moveUp = h - yh
    moveDown = yl - l
    if 0 < moveUp > moveDown:
        PDM = moveUp
    else:
        PDM = 0
        
    return PDM

def NDM(o, h, l, c, yo, yh, yl, yc):
    moveDown = yl - l
    moveUp = h - yh
    if 0 < moveDown > moveUp:
        NDM = moveDown
    else:
        NDM = 0
    
    return NDM

def RSI(x):
    up, down = [i for i in x if i > 0], [i for i in x if i <= 0]
    if len(down) == 0:
        return 100
    elif len(up) == 0:
        return 0
    else:
        up_average = sum(up) / len(up)
        down_average = - sum(down) / len(down)
        return 100 * up_average / (up_average + down_average)
    
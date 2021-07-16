# import os
# import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# import joblib

def create_data(df):
    # set return and direction (label)
    df['return'] = 10*np.log(df['Adj Close'].shift(-5) / df['Adj Close'])

    # simple moving average
    df['sma10'] = df['Adj Close'].rolling(30).mean()

    # Commodity Channel Index in 24 days
    df['typical-price'] = (df['High'] + df['Low'] + df['Close']) / 3
   

    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    df['adosc-ema3'] = df['adosc'].ewm(span=3, adjust=False).mean()
    df['adosc-ema10'] = df['adosc'].ewm(span=10, adjust=False).mean()
    df['adosc-SG'] = np.where((df['adosc-ema3'] - df['adosc-ema10']) > 0, 1, -1)

     # Commodity Channel Index in 24 days
    df['typical-price'] = (df['High'] + df['Low'] + df['Close']) / 3
   
    

    # simple moving average
    for i in [20,30,40,45,60,90,120,150,180,200,210,240,270,300]:
        df['sma'+str(i)] = df['Adj Close'].rolling(i).mean()
        df['sma'+str(i)] = (df['sma'+str(i)] / df['sma'+str(i)].shift(1)-1)*100

    for i in [20,30,40,45,60,90,120,150,180,200,210,240,270,300]:
        df['today_by_sma'+str(i)+'ratio']= df['Adj Close']/df['sma'+str(i)] 
#         df['sma'+str(i)] = Zero_One_Scale(df['sma'+str(i)])
    for i in [10,30,60,90,120,150,180,200,210,240,270,300]:
        df["ema"+str(i)]=df["Adj Close"].ewm(span=i).mean()
        df["ema"+str(i)] = (df["ema"+str(i)] / df["ema"+str(i)].shift(1))*100
#         df["ema"+str(i)] = Zero_One_Scale(df["ema"+str(i)])
  
    
    for i in range(15,150,30):
        for k in range(30,270,60):
            df['ratio_sma'+str(k)] = df['Adj Close'].rolling(k).mean()
            df['ratio_sma'+str(i)] = df['Adj Close'].rolling(i).mean()
            df['ratio_sma'+str(i)+'_'+str(k)] = (df['ratio_sma'+str(i)] / df['ratio_sma'+str(k)]-1)*10
#             df['ratio_sma'+str(i)+'_'+str(k)] = Zero_One_Scale(df['ratio_sma'+str(i)])

    for i in [5,10,15,20,45,61,81,121,161]:
        df['Highest_in_range'+str(i)] = df['Adj Close'].rolling(window=i).max()
        df['Highest'+str(i)+'ago'] = (df['Highest_in_range'+str(i)].shift())
        for m in [30,81,90,150,60,60]:    
            df['Highest'+str(i)+','+str(m)+'days_ago'] = df['Adj Close'] / df['Highest'+str(i)+'ago'].shift(m)
#             df['Highest'+str(i)+','+str(m)+'days_ago'] = Zero_One_Scale(df['Highest'+str(i)+','+str(m)+'days_ago'])
        
        #今日の終値が過去何日間の高音に対してどの程度あるか
        df['Highest'+str(i)] = df['Adj Close'] / df['Highest_in_range'+str(i)]
#         df['Highest'+str(i)] = Zero_One_Scale(df['Highest'+str(i)])

    # i日前からの価格の変動率_Dena
    for i in [5,10,15,20,40,61,81,121,161]:
        df['Adjclose_today_'+str(i)+'daysago_ratio'] = df['Adj Close'] / df['Adj Close'].shift(i)

    # 高値と安値の比率の対数
    df['high_low_ratio'] = np.log(df['High'] / df['Low'])
    # 高値と安値の比率の対数の移動平均
    for i in [20,30,40,45,60,90,120,150,180,200,210,240,270,300]:
        df['high_low_ratio_'+str(i)+'_sma'] = df['high_low_ratio'].rolling(i).mean()
        df['high_low_ratio_'+str(i)+'_sma_ratio'] = (df['high_low_ratio_'+str(i)+'_sma'] / df['high_low_ratio_'+str(i)+'_sma'].shift(1)-1)*100

    for i in [20,30,40,45,60,90,120,150,180,200,210,240,270,300]:
        df['today_by_high_low_ratio_'+str(i)+'_sma_ratio']= df['Adj Close']/df['high_low_ratio_'+str(i)+'_sma'] 

    # i日前からの出来高の変動率_Dena
    for i in [5,10,15,20,40,61,81,121,161]:
        df['Volume_'+str(i)+'daysago_ratio'] = df['Volume'] / df['Volume'].shift(i)
 

    


    return df

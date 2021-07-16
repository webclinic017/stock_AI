import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    model_stock = 'sandp500'
    stock = 'nasdaq100'
    model = load_model('../model/{}/model_2.h5'.format(model_stock))
    #全時系列データの呼び出し
    path = '../../../stock_data/time_series_data/{}'.format(stock)
    stock_datas = os.listdir(path)
        
        
    position = pd.DataFrame()
    
    for i in stock_datas:
        print(i)
       
        stock_data = pd.read_csv('../../stock_data/time_series_data/{}/'.format(stock) + i)
        df = stock_data.copy()
        df = df.set_index('Date')
        #範囲を多めに取って計算量を減らす
        df = df['2010-01-01':'2021-04-09']
        
#         try:
        feature1(df)
#             print(df)
        cols = ['sma30','sma60', 'sma90','sma120' ,'sma150','sma180','sma210','sma240','sma270'

                ,'sma75_30', 'sma75_90', 'sma75_150', 'sma75_210', 'sma105_30', 'sma105_90',
                'sma105_150', 'sma105_210', 'sma135_30', 'sma135_90', 'sma135_150', 'sma135_210'

            ,'Highest81','Highest81,30days_ago','Highest81,60days_ago','Highest121','Highest121,30days_ago'
                ,'Highest121,60days_ago','Highest161','Standard_deviation_normalization30',
                'Standard_deviation_normalization35','Standard_deviation_normalization40'
            ,'adosc']
        std_model = StandardScaler()
        df[cols] = std_model.fit_transform(df[cols])
        # print(df[cols])

        
        
        
        df['predict'] = model.predict(df[cols])
        # print(df[i])
        data = df['2020-01-01':'2021-04-09']
        print(data['predict',"validation"])
        position = pd.concat([position ,data['predict',"validation"]],axis = 0)
        print(position)

#         except:
#             continue

    position.to_csv('result/model_{}/{}_scatter.csv'.format(model_stock,stock))
    print(position)
    sns.scatterplot(x=position['predict'], y=position["validation"])
    
    


def feature1(df):
    
    # simple moving average
    for i in range(30,300,30):
        df['sma'+str(i)] = df['Adj Close'].rolling(i).mean()
        df['sma'+str(i)] = df['sma'+str(i)] / df['sma'+str(i)].shift(1)


    for i in range(15,150,30):
        for k in range(30,270,60): 
            df['sma'+str(k)] = df['Adj Close'].rolling(k).mean()
            df['sma'+str(i)] = df['Adj Close'].rolling(i).mean()
            df['sma'+str(i)+'_'+str(k)] = df['sma'+str(i)] / df['sma'+str(k)]

    for term in range(5,50,5):
        df['SMA'+str(term)] = df['Adj Close'].rolling(term).mean()
        df['STD'+str(term)] = df['Adj Close'].rolling(term).std()
        df['Standard_deviation_normalization'+str(term)] = 100 * 2 * df['STD'+str(term)] / df['SMA'+str(term)]

    for i in range(41,201,40):
        df['Highest'+str(i)] = df['Adj Close'].rolling(window=81).max()
        df['Highest'+str(i)] = df['Highest'+str(i)].shift()
        for m in range(30,90,30):
            df['Highest'+str(i)+','+str(m)+'days_ago'] = df['Adj Close'] / df['Highest'+str(i)].shift(m)

        #今日の終値が過去何日間の高音に対してどの程度あるか
        df['Highest'+str(i)] = df['Adj Close'] / df['Highest'+str(i)]


    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    validation(df)
    df.dropna(inplace=True)

    return df

def validation(df):
    k=10
    df["Open"] = df["Open"].shift(-1)
    df["validation"] = np.nan
    for i in range(0,len(df)-k):
        
        df["validation"][i] = df['Adj Close'][i+k]/df['Open'][i]

    return df



def Normalize(df):
    df_normalized = (df - df.mean(axis=0)) / df.std(axis=0)

def One_One_Scale(df):
    df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_scaled
        
main()
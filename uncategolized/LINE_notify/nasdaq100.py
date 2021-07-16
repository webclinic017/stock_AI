
import pandas_datareader as web
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpl
import numpy as np
import requests
import datetime as dt
import os
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model



def main():
    model_stock = 'sandp500'
    stock = 'nasdaq100'
    model = load_model('../machine_learning/model4/model/{}/model1.h5'.format(model_stock))
    
    
    df = pd.read_csv('../stock_data/stock_code/{}.csv'.format(stock))
    print(df)

    signs = []
    for symbol in df['symbol']:
        signs.append(symbol)


    # content = '\n\nToday`s stock report' + str(dt.date.today())
    count = 0
    position = pd.DataFrame()

    # #----------------------------------------------------------------------------
    for i in signs :   
        start=dt.datetime.now()-dt.timedelta(days=400)
        end=dt.datetime.now()
        count += 1
        print(count)
        print(i)
        try:
            df = yf.download(i, start, end, interval='1d')

            feature1(df)
            print(df)
            cols = ['sma30','sma60', 'sma90','sma120' ,'sma150','sma180','sma210','sma240','sma270'

                    ,'sma75_30', 'sma75_90', 'sma75_150', 'sma75_210', 'sma105_30', 'sma105_90',
                    'sma105_150', 'sma105_210', 'sma135_30', 'sma135_90', 'sma135_150', 'sma135_210'

                ,'Highest81','Highest81,30days_ago','Highest81,60days_ago','Highest121','Highest121,30days_ago'
                    ,'Highest121,60days_ago','Highest161','Standard_deviation_normalization30',
                    'Standard_deviation_normalization35','Standard_deviation_normalization40'
                ,'adosc']
            std_model = StandardScaler()
            df[cols] = std_model.fit_transform(df[cols])
            print(df[cols])
        
            df[i] = model.predict(df[cols])
            print(df[i])
            data = df['2016-01-01':'2021-04-09']
            position = pd.concat([position ,data[i]],axis = 1)
            # print(position)

        except:
            continue
    

    series = position.iloc[-1]
    print(series)
    series = series.sort_values(ascending=False) 
    code_list = series.index.values.tolist()
    point_list = series.values.tolist()
#             どのくらい抽出するのか
    pickup = 15
    print(code_list[0:pickup])
    print(point_list[0:pickup])

    content = [code_list,point_list]
    send_line_notify(content)





            
        



def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'GlXSB9BURpdoAifXsABCBhAyNoehXTZ6Gp7Z2E8Wc2I'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)
    

def feature1(df):
     # set return and direction (label)
    df['return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['direction'] = np.where(df['return'] > 0, 1, -1)
    df['direction'] = df['direction'].shift(-1)
    df['return'] = One_One_Scale(df['return'])
    df['return'] = df['return'].shift(-1)
    
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

    df.dropna(inplace=True)

    return df

def Normalize(df):
    df_normalized = (df - df.mean(axis=0)) / df.std(axis=0)

def One_One_Scale(df):
    df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_scaled
        
main()
    
#     # high in the past 81 days
#     df['Highest81'] = df['Adj Close'].rolling(window=81).max()
#     # 標準偏差を計算
#     try:
#         short_sma = 10
#         df['SMA'+str(short_sma)] = df['Adj Close'].rolling(window=short_sma).mean()
#         df['STD'] = df['Adj Close'].rolling(window=25).std()
#         df['Standard_deviation_normalization'] = 100 * 2 * df['STD'] / df['SMA'+str(short_sma)]


#         highest = df['Highest81'][-1]
#         highest_2 = df['Highest81'][-60]
#         close = df['Adj Close'][-1]
#         std = df['Standard_deviation_normalization'][-1]
        
#     except :
#         continue
 
#     #今日の終値が昔の高値よりかなり大きく、上昇トレンドが形成されている
#     if close > 1.15 * highest_2 and std < 5 and close > 0.96 * highest:

#         print(highest)
#         print('signal first')

#         print('signal')
#         content_x = "\n{} 👍 \n逆指値を入れる値段: ¥{}\n前の終値: ¥{}".format(symbol, round(highest, 5), round(close, 5))
#         content += '\n'+content_x

#     else :
#         print('NO')
# print(content)
# send_line_notify(content)









import pandas as pd
import pandas_datareader as web
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpl
import numpy as np
import requests
import datetime as dt


def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'GlXSB9BURpdoAifXsABCBhAyNoehXTZ6Gp7Z2E8Wc2I'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)
    
df = pd.read_csv('../symbols/rakuten_usstock.csv', encoding='shift_jis')
print(df)

signs = []
for symbol in df['symbol']:
    signs.append(symbol)


content = '\n\nToday`s stock report' + str(dt.date.today())
count = 0

#----------------------------------------------------------------------------
for symbol in signs[1500:3000] :   

    start=dt.datetime.now()-dt.timedelta(days=400)
    end=dt.datetime.now()
    count += 1
    print(count)
    
    
    try:
        df = yf.download(symbol, start, end, interval='1d')
        market_cap = web.get_quote_yahoo(symbol)['marketCap']
        cap = int(str(market_cap).split()[1])
      
        
        print(symbol)
        print('時価総額は'+ str(cap))
#         type(market_cap)
    except:
        print('時価総額なし'+str(symbol))

    

    if df.empty :
        continue

    
    
    # high in the past 81 days
    df['Highest81'] = df['Adj Close'].rolling(window=81).max()
    # 標準偏差を計算
    try:
        short_sma = 10
        df['SMA'+str(short_sma)] = df['Adj Close'].rolling(window=short_sma).mean()
        df['STD'] = df['Adj Close'].rolling(window=25).std()
        df['Standard_deviation_normalization'] = 100 * 2 * df['STD'] / df['SMA'+str(short_sma)]


        highest = df['Highest81'][-1]
        highest_2 = df['Highest81'][-60]
        close = df['Adj Close'][-1]
        std = df['Standard_deviation_normalization'][-1]
        
    except :
        continue
 
    #今日の終値が昔の高値よりかなり大きく、上昇トレンドが形成されている
    if close > 1.15 * highest_2 and std < 5 and close > 0.96 * highest:

        print(highest)
        print('signal first')

        print('signal')
        content_x = "\n{} 👍 \n逆指値を入れる値段: ¥{}\n前の終値: ¥{}".format(symbol, round(highest, 5), round(close, 5))
        content += '\n'+content_x

    else :
        print('NO')
print(content)
send_line_notify(content)







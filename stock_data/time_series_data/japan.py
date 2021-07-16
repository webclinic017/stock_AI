import pandas as pd
import pandas_datareader as web
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpl
import numpy as np
import requests
import datetime as dt
   


df = pd.read_csv('../../stock_data/stock_code/japan_all.csv')
# print(df)
df2 = df[df['type'] != 'ETF・ETN']
# print(df2)

symbols = []
content = '\n\nToday`s stock report' + str(dt.date.today())
count = 0

#----------------------------------------------------------------------------
for symbol in df2['symbol'][1:1500] :   
    
        x = str(symbol) + '.T'
#         print(x)
        symbols.append(x)
        
# print(symbols)


# #----------------------------------------------------------------------------

for symbol in symbols:
    count += 1
    print(count)
    
    
    try:
        df = yf.download(symbol,interval='1d')
        print(df)
        
    except:
        continue

    df.to_csv('japan_all/{}.csv'.format(symbol))
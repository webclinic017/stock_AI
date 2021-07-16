
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpf


# import financial data
# you can search your favorite company code here >> https://finance.yahoo.com/lookup/ 
start = '1990-01-01'
end = dt.datetime.now()
stock_code = '^GSPC' 
df = yf.download(stock_code, start, end, interval='1d') 

percentChange = []

weekdays=[0,1,2,3,4]
for weekday in weekdays:
    weekchange=[]
    position = 0
    for i in df.index:
        
        counter = 0
        if weekday == i.weekday():
            if position == 0:
                buy_price=df['Open'][i]    
                position=1
       
        elif weekday+1 == i.weekday() or weekday+1==5:
            if position == 1:
                sell_price=df['Open'][i]
                percent=(sell_price/buy_price-1)*100
                weekchange.append(percent)
                counter += 1
                position=0
           

    percentChange.append(weekchange)
   


youbi=['月曜日','火曜日','水曜日','木曜日','金曜日']
count=0
for weekchange in percentChange:

    gains = 0
    numgains = 0
    losses = 0
    numlosses = 0
    total_return = 1
    for i in weekchange:
        if i>0:
            gains += i
            numgains += 1
        else:
            losses += i
            numlosses += 1
        
        total_return=total_return*((i/100)+1)
    print('{}から現在まで'.format(start))
    print(youbi[count])
    print('トータルリターンは{} %'.format(round(100*(total_return-1),2)))
    print('平均リターンは'+str(round(gains/numgains,2))+'%')
    print('平均損失は'+str(round(losses/numlosses,2))+'%')
    print('勝率は'+str(100*numgains/(numgains+numlosses))+'%')
    print('-------------------------------------------')

    count += 1






import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpf

start = '1990-01-01'
end = '2021-02-01'
stock_code = '^GSPC' 
df = yf.download(stock_code, start, end, interval='1d') 
term = 60

# chek the highest price in the past {term} times
df['Highest'+str(term)] = df.iloc[:, 1].rolling(window=term).max()
# chek the highest price in the past {term} times
df['Lowest'+str(term)] = df.iloc[:, 2].rolling(window=term).min()


#買い戦略

position=0
count=0
percentChange = []
for i in range(1, len(df)):
    Highest = df['Highest'+str(term)][i-1]
    Lowest = df['Lowest'+str(term)][i-1]
    high_price = df['High'][i]
    low_price = df['Low'][i]
    close = df['Adj Close'][i]
    open_price = df['Open'][i]
    last_close = df['Close'][i-1]
    last_high = df['High'][i-1]
    

    # avoid NaN data 
    if np.isnan(Highest):
        continue

    else:
        
        if (high_price > Highest):
          
    
            if (position == 0):
                position = 1
            
                #print('シグナル点灯!!!!! ')

        elif (open_price > last_high):
            if (close < last_close):
                if (position == 1):
                    position = 2
                    sell_price = close
                    #print("Sell at the price {}".format(sell_price))

        if (position==2 and high_price > sell_price*1.06):
    
            position = 0
            buy_price = high_price
            #print('Buy at the price {}'.format(buy_price))
            percent = -(buy_price / sell_price - 1) * 100
            percentChange.append(percent)

        elif (position==2 and open_price < 0.95*sell_price):
            
            position = 0
            buy_price = low_price
            #print('Buy at the price {}'.format(buy_price))
            percent = -(buy_price / sell_price - 1) * 100
            percentChange.append(percent)

    count += 1

#print(percentChange)

# statistic
gains = 0
numgains = 0
losses = 0
numlosses = 0
total_return = 1


for i in percentChange:
    if i > 0:
        numgains += 1
        gains += i
    else:
        numlosses += 1
        losses += i
    total_return = total_return * ((i / 100) + 1)



            
print('{}から現在まで'.format(start))
print('トータルリターンは{} %'.format(round(100*(total_return-1),2)))
print('平均リターンは'+str(round(gains/numgains,2))+'%')
print('平均損失は'+str(round(losses/numlosses,2))+'%')
print('勝率は'+str(100*numgains/(numgains+numlosses))+'%')
print('-------------------------------------------')





            



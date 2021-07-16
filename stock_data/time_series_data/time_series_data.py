import pandas as pd
import yfinance as yf


stock = 'sandp500'
df = pd.read_csv('../stock_code/{}.csv'.format(stock))
print(df)

for i in df['symbol']:
#     symbol = str(i) + '.T'
    start = '2020-01-01'
    end = '2021-04-01'
#     data = yf.download(i, start, end, interval='60m') 
    data = yf.download(i, interval='1d')
    print(data)
    data.to_csv('{}/{}.csv'.format(stock,i))
    
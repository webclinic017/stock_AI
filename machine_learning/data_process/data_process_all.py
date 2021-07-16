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




    df['amount'] = df['Adj Close'] * df['Volume']

    # simple moving average
    df['sma10'] = df['Adj Close'].rolling(30).mean()
    df['sma10-FP'] = (df['sma10'] - df['sma10'].shift(1)) / df['sma10'].shift(1)
 
    # Moving Average Convergence Divergence
    df['macd'] = df['Adj Close'].rolling(12).mean() - df['Adj Close'].rolling(26).mean()
    df['macd-SG'] = df['macd'].rolling(9).mean()
    df['macd-histogram'] = df['macd'] - df['macd-SG']
    df['macd-histogram'] = np.where(df['macd-histogram'] > 0, 1, -1)
    df['macd-SG'] = np.where(df['macd-SG'] > 0, 1, -1)
    df['macd'] = np.where(df['macd'] > 0, 1, -1)
    # Commodity Channel Index in 24 days
    df['typical-price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['sma-cci'] = df['typical-price'].rolling(24).mean()
    df['mean-deviation'] = np.abs(df['typical-price'] - df['sma-cci'])
    df['mean-deviation'] = df['mean-deviation'].rolling(24).mean()
    df['cci'] = (df['typical-price'] - df['sma-cci']) / (0.015 * df['mean-deviation'])
    df['cci-SG'] = np.where(df['cci'] > 0, 1, -1)
    # MTM 10
    df['mtm10'] = df['Adj Close'] - df['Adj Close'].shift(10)
    df['mtm10'] = np.where(df['mtm10'] > 0, 1, -1)
    # Rate of Change in 10 days
    df['roc'] = (df['Adj Close'] - df['Adj Close'].shift(10)) / df['Adj Close'].shift(10)
    df['roc-SG'] = np.where(df['roc'] > 0, 1, -1)
    df['roc-FP'] = (df['roc'] - df['roc'].shift(1))

    # Relative Strength Index in 5 days
    df['price-change'] = df['Adj Close'] - df['Adj Close'].shift(1)
    df['price-change-percentage'] = df['Adj Close'] / df['Adj Close'].shift(1)
    df['rsi'] = df['price-change'].rolling(5).apply(RSI) / 100
    df['rsi-FP'] = (df['rsi'] - df['rsi'].shift(1))

    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    df['adosc-ema3'] = df['adosc'].ewm(span=3, adjust=False).mean()
    df['adosc-ema10'] = df['adosc'].ewm(span=10, adjust=False).mean()
    df['adosc-SG'] = np.where((df['adosc-ema3'] - df['adosc-ema10']) > 0, 1, -1)

    # AR 26
    hp_op = (df['High'] - df['Open']).rolling(26).sum()
    op_lp = (df['Open'] - df['Low']).rolling(26).sum()
    df['ar26'] = hp_op / op_lp

    # BR 26
    hp_cp = (df['High'] - df['Close']).rolling(26).sum()
    cp_lp = (df['Close'] - df['Low']).rolling(26).sum()
    df['br26'] = hp_cp / cp_lp

    # VR 26
    
    # BIAS 20
    sma20 = df['Adj Close'].rolling(20).mean()
    df['bias20'] = (df['Adj Close'] - sma20) / sma20
    df['bias20'] = np.where(df['bias20'] > 0, 1, -1)
    
    
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
                
  
    OBV = []
    OBV.append(0)
    for i in range(1, len(df.Close)):
        if df.Close[i] > df.Close[i-1]: #If the closing price is above the prior close price 
              OBV.append(OBV[-1] + df.Volume[i]) #then: Current OBV = Previous OBV + Current Volume
        elif df.Close[i] < df.Close[i-1]:
              OBV.append( OBV[-1] - df.Volume[i])
        else:
              OBV.append(OBV[-1])
                
    #Store the OBV and OBV EMA into new columns
    df['OBV'] = OBV
    df['OBV_EMA'] = df['OBV'].ewm(com=20).mean()   
    df['OBV_ratio'] = df['OBV'] / df['OBV_EMA']
    
    df['OBV_EMA_ratio'] =  df['OBV_EMA'] / df['OBV_EMA'].shift(1)
                
#     df['OBV'] = Zero_One_Scale(df['OBV'])
#     df['OBV_EMA'] = Zero_One_Scale(df['OBV_EMA'])
#     df['OBV_ratio'] = Zero_One_Scale(df['OBV_ratio'])
#     df['OBV_EMA_ratio'] = Zero_One_Scale(df['OBV_EMA_ratio'])

     # Commodity Channel Index in 24 days
    df['typical-price'] = (df['High'] + df['Low'] + df['Close']) / 3
   
    

    # simple moving average
    for i in [30,45,60,90,120,150,180,200,210,240,270,300]:
        df['sma'+str(i)] = df['Adj Close'].rolling(i).mean()
        df['sma'+str(i)] = (df['sma'+str(i)] / df['sma'+str(i)].shift(1)-1)*10
#         df['sma'+str(i)] = Zero_One_Scale(df['sma'+str(i)])
    for i in [30,60,90,120,150,180,200,210,240,270,300]:
        df["ema"+str(i)]=df["Adj Close"].ewm(span=i).mean()
        df["ema"+str(i)] = (df["ema"+str(i)] / df["ema"+str(i)].shift(1))*10
#         df["ema"+str(i)] = Zero_One_Scale(df["ema"+str(i)])
    
    
    #calucurate aroon
    for periods in [14,20]:
        df['aroon_up'+str(periods)] = df['High'].rolling(periods+1).apply(lambda x: x.argmax(), raw=True) / periods * 100
        df['aroon_down'+str(periods)] = df['Low'].rolling(periods+1).apply(lambda x: x.argmin(), raw=True) / periods * 100
        df['aroon_ratio'+str(periods)] = df['aroon_up'+str(periods)] / df['aroon_down'+str(periods)]
#         df['AROONOSC'+str(periods)] = df['aroon_up'+str(periods)] - df['aroon_down'+str(periods)]

#         df['AROONOSC'+str(periods)] = Zero_One_Scale(df['AROONOSC'+str(periods)])


     #calucurate ADX
    df["TrueRange"] = np.nan
    df["PDM"] = np.nan
    df["NDM"] = np.nan
    for i in range(1,len(df)):
        df["TrueRange"][i] = TrueRange(df["Adj Close"][i],df["High"][i],df["Low"][i],df["Open"][i],df["Adj Close"][i-1])
        df["PDM"][i] = PDM(df["Open"][i],df["High"][i],df["Low"][i],df["Adj Close"][i],df["Open"][i-1],df["High"][i-1],df["Low"][i-1],df["Adj Close"][i-1])
        df["NDM"][i] = NDM(df["Open"][i],df["High"][i],df["Low"][i],df["Adj Close"][i],df["Open"][i-1],df["High"][i-1],df["Low"][i-1],df["Adj Close"][i-1])
    
    df['PDI'] = df["PDM"].rolling(14).sum()/df["TrueRange"].rolling(14).sum() * 100
    df['NDI'] = df["NDM"].rolling(14).sum()/df["TrueRange"].rolling(14).sum() * 100

    
    df['DX'] = (df['PDI']-df['NDI']).abs()/(df['PDI']+df['NDI']) * 100
    df['DX'] = df['DX'].fillna(0)
    
    df['ADX'] = df['DX'].rolling(14).mean()
    df['ADXR'] = df['ADX'].rolling(14).mean()
    
#     df['ADX'] = Zero_One_Scale(df['ADX'])
#     df['ADXR'] = Zero_One_Scale(df['ADXR'])
    
    
            
    for term in range(5,50,5):
        df['SMA'+str(term)] = df['Adj Close'].rolling(term).mean()
        df['STD'+str(term)] = df['Adj Close'].rolling(term).std()
        df['Standard_deviation_normalization'+str(term)] = 100 * 2 * df['STD'+str(term)] / df['SMA'+str(term)]
#         df['Standard_deviation_normalization'+str(term)] = Zero_One_Scale(df['Standard_deviation_normalization'+str(term)])
        

        
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
    

def data_process(save_path):
    count = 0
    for i in stock_datas:
        count = count +1
        print(count)
        print(i)
        try:
            df = pd.read_csv('../../stock_data/time_series_data/{}/'.format(model_stock) + i)
            df = df.set_index('Date')
            #範囲を多めに取って計算量を減らす
            df = df['2020-01-01':'2021-3-31']
            df = create_data(df)
            # print(df)
            df = df['2021-01-06':'2021-3-31']
            # print(df.isnull().any())
            data = data.append(df)

        except:
            continue
    # drop row contains NaN
    data.dropna(inplace=True)
    data.to_csv(save_path)






    
 

    


    return df

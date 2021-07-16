import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import datetime

class myclass():
    def __init__(self, model_stock):
        self.model_stock = model_stock

    
    def main(self):
        
        #全時系列データの呼び出し
        path = '../../../stock_data/time_series_data/{}'.format(self.model_stock)
        stock_datas = os.listdir(path)
        mi_score = pd.DataFrame()
        count = 0

        
        input_data = []
        output_data = []
        for i in stock_datas:
            count = count +1
            print(count)
            print(i)
            # try:
            df = pd.read_csv('../../../stock_data/time_series_data/{}/'.format(self.model_stock) + i)
            df = df.set_index('Date')

            #範囲を多めに取って計算量を減らす
            df = df['2012-01-01':'2021-03-01']
            feature1(df)
            df = df['2015-01-01':'2020-01-01']  
            cols = ['sma30','sma60', 'sma90','sma120' ,'sma150','sma180','sma210','sma240','sma270'

                    ,'ratio_sma75_30', 'ratio_sma75_90', 'ratio_sma75_150', 'ratio_sma75_210', 'ratio_sma105_30', 'ratio_sma105_90',
                    'ratio_sma105_150', 'ratio_sma105_210', 'ratio_sma135_30', 'ratio_sma135_90', 'ratio_sma135_150', 'ratio_sma135_210'

                    ,'Highest81','Highest81,30days_ago','Highest81,60days_ago','Highest121','Highest121,30days_ago'
                    ,'Highest121,60days_ago','Highest161','Standard_deviation_normalization30',
                    'Standard_deviation_normalization35','Standard_deviation_normalization40'
                    ,'adosc']
            
            print(df)
            print(df.isna().sum())


            # print(df)
        
            for n in range(0,len(df.index)-70):
                df1 = df[cols][n:n+64]
                data = df1.values
            
                print(data)
                data_expanded = np.expand_dims(data,axis=0)
                print(data_expanded.shape)
                input_data.append(data_expanded)
                

                data2 = df['validation'][n+64]
                print(data2.shape)
                # data_expanded2 = np.expand_dims(data2,axis=0)
                # print(data_expanded2.shape)
                
                output_data.append(data2)

                    
            # except:
            #     continue
        input_data = np.concatenate(input_data ,axis=0)
        input_data[np.isnan(input_data)] = 0.5
        # input_data = np.array(input_data)
        output_data = np.array(output_data)
        output_data[np.isnan(output_data)] = 0.5

        print(input_data.shape)
        print(output_data.shape)
        print(input_data)


        # train_X = np.reshape(input_data, (-1, 10, 1))
        # train_Y = np.reshape(output_data, (-1, 1, 1))
        # print(train_X.shape)
        # print(train_Y.shape)

        # 書き込み
        np.save('preprocess_data/{}_train_X.npy'.format(self.model_stock),input_data )
        np.save('preprocess_data/{}_train_Y.npy'.format(self.model_stock),output_data)
        print('ok')


def feature1(df):
#     date = df.index
#     for i in range(len(df)):
#         date = df.index[i]
#         date_dt = datetime.datetime.strptime(date, '%Y-%m-%d ')
#         print(date_dt)
     #     calucurate A/D
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
                
    df['A/D'] = Zero_One_Scale(df['A/D'])
    df['A/D_EMA'] = Zero_One_Scale(df['A/D_EMA'])
    df['A/D_ratio'] = Zero_One_Scale(df['A/D_ratio'])
    df['A/D_EMA_ratio'] = Zero_One_Scale(df['A/D_ratio'])
    df['ADOSC'] = Zero_One_Scale(df['ADOSC'])
   
    
    
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
                
    df['OBV'] = Zero_One_Scale(df['OBV'])
    df['OBV_EMA'] = Zero_One_Scale(df['OBV_EMA'])
    df['OBV_ratio'] = Zero_One_Scale(df['OBV_ratio'])
    df['OBV_EMA_ratio'] = Zero_One_Scale(df['OBV_EMA_ratio'])
    

    # simple moving average
    for i in [30,60,90,120,150,180,200,210,240,270,300]:
        df['sma'+str(i)] = df['Adj Close'].rolling(i).mean()
        df['sma'+str(i)] = df['sma'+str(i)] / df['sma'+str(i)].shift(1)
        df['sma'+str(i)] = Zero_One_Scale(df['sma'+str(i)])
    for i in [30,60,90,120,150,180,200,210,240,270,300]:
        df["ema"+str(i)]=df["Adj Close"].ewm(span=i).mean()
        df["ema"+str(i)] = df["ema"+str(i)] / df["ema"+str(i)].shift(1)
        df["ema"+str(i)] = Zero_One_Scale(df["ema"+str(i)])
    
    
    #calucurate aroon
    for periods in [14,20]:
        df['aroon_up'+str(periods)] = df['High'].rolling(periods+1).apply(lambda x: x.argmax(), raw=True) / periods * 100
        df['aroon_down'+str(periods)] = df['Low'].rolling(periods+1).apply(lambda x: x.argmin(), raw=True) / periods * 100
        df['aroon_ratio'+str(periods)] = df['aroon_up'+str(periods)] / df['aroon_down'+str(periods)]
        df['AROONOSC'+str(periods)] = df['aroon_up'+str(periods)] - df['aroon_down'+str(periods)]

        df['aroon_up'+str(periods)] = Zero_One_Scale(df['aroon_up'+str(periods)])
        df['aroon_down'+str(periods)] = Zero_One_Scale(df['aroon_down'+str(periods)])
        df['aroon_ratio'+str(periods)] = Zero_One_Scale(df['aroon_up'+str(periods)])
        df['AROONOSC'+str(periods)] = Zero_One_Scale(df['AROONOSC'+str(periods)])


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
    
    df['ADX'] = Zero_One_Scale(df['ADX'])
    df['ADXR'] = Zero_One_Scale(df['ADXR'])
    
    
    for i in range(15,150,30):
        for k in range(30,270,60):
            df['ratio_sma'+str(k)] = df['Adj Close'].rolling(k).mean()
            df['ratio_sma'+str(i)] = df['Adj Close'].rolling(i).mean()
            df['ratio_sma'+str(i)+'_'+str(k)] = df['ratio_sma'+str(i)] / df['ratio_sma'+str(k)]
            df['ratio_sma'+str(i)+'_'+str(k)] = Zero_One_Scale(df['ratio_sma'+str(i)])

            
    for term in range(5,50,5):
        df['SMA'+str(term)] = df['Adj Close'].rolling(term).mean()
        df['STD'+str(term)] = df['Adj Close'].rolling(term).std()
        df['Standard_deviation_normalization'+str(term)] = 100 * 2 * df['STD'+str(term)] / df['SMA'+str(term)]
        df['Standard_deviation_normalization'+str(term)] = Zero_One_Scale(df['Standard_deviation_normalization'+str(term)])
        
    for i in [15,45,81,121,161]:
        df['Highest'+str(i)] = df['Adj Close'].rolling(window=81).max()
        df['Highest'+str(i)] = df['Highest'+str(i)].shift()
        for m in [30,90,150,60,60]:
            df['Highest'+str(i)+','+str(m)+'days_ago'] = df['Adj Close'] / df['Highest'+str(i)].shift(m)
            df['Highest'+str(i)+','+str(m)+'days_ago'] = Zero_One_Scale(df['Highest'+str(i)+','+str(m)+'days_ago'])
        
        #今日の終値が過去何日間の高音に対してどの程度あるか
        df['Highest'+str(i)] = df['Adj Close'] / df['Highest'+str(i)]
        df['Highest'+str(i)] = Zero_One_Scale(df['Highest'+str(i)])
        
    # ADOSC
    df['adosc'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
    df['adosc'] = df['adosc'].cumsum()
    df['adosc'] = Zero_One_Scale(df['adosc'])
        


    validation(df)
    

    return df




def validation(df):
    k=5
    df["Open"] = df["Open"].shift(-1)
    df["validation"] = np.nan
    for i in range(0,len(df)-k):

        df["validation"][i] = df['Adj Close'][i+k]/df['Open'][i]

    df["validation"] = Zero_One_Scale(df["validation"])

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



def Normalize(df):
    df_normalized = (df - df.mean(axis=0)) / df.std(axis=0)

def One_One_Scale(df):
    df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_scaled

def Zero_One_Scale(df):
    df_scaled = (df - df.min()) / (df.max() - df.min())
    return df_scaled

myclass = myclass('nasdaq100')
myclass.main()


import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import operator
# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style

import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter

import statistics
# N-Lags model check
import yfinance as yf
# 他ファイルのインポート・リロード------------------------
import imp

def prediction_csv(model,model_name,cols):


  #全時系列データの呼び出し
  model_stock = 'sandp500'
  path = '/content/drive/MyDrive/stock_data/stock_code/{}.csv'.format(model_stock)
  data = pd.read_csv(path)
  print(data)




  # # 作業ディレクトリの移動と確認------------------------
  os.chdir('/content/drive/MyDrive/machine_learning/data_preprocess/')
  # !ls
  # 他ファイルのインポート・リロード------------------------
  import data_process
  imp.reload(data_process)

  position = pd.DataFrame()
  count = 0
  for  file in data['symbol'] :
    count = count +1
    print(count)
    print(file)
    try:
      df = yf.download(file, start='2019-01-01')
      df = data_process.create_data(df)
      print(df)
      df[file] = model.predict(df[cols])
      print(df[file])
      position = pd.concat([position ,df[file] ],axis = 1)

    except: 
      print('errror')
      continue
  position.to_csv('/content/drive/MyDrive/machine_learning/light_gbm/backtest/prediction_{}.csv'.format(model_name))




def result1(percentChange):
    gains = 0
    numGains = 0
    losses = 0
    numLosses = 0
    for i in percentChange:
        if i > 0:
            gains += i
            numGains += 1
        elif i < 0:
            losses += i
            numLosses += 1
     

    total_return = statistics.mean(percentChange)

    if numGains > 0:
        average_gain = gains / numGains
        max_return = max(percentChange)
    else:
        average_gain = np.nan
        max_return = np.nan

    if numLosses > 0:
        average_loss = losses / numLosses
        max_loss = min(percentChange)
    else:
        average_loss = np.nan
        max_loss = np.nan

    if numGains > 0 and numLosses > 0:
        risk_reward_ratio = - average_gain / average_loss

    elif numGains == 0 and numLosses > 0:
        risk_reward_ratio = 0

    elif numGains > 0 and numLosses == 0:
        risk_reward_ratio = average_gain

    else:
        risk_reward_ratio = np.nan

    if numGains > 0 or numLosses > 0:
        batting_ave = numGains / (numGains + numLosses)
    else:
        batting_ave = np.nan


    return [total_return, average_gain, average_loss, max_return, max_loss, risk_reward_ratio, batting_ave]


def show_results(results):
    labels = ['average Return', 'Average Gain', 'Average Loss', 'Max Return', 'Max Loss', 'Risk Reward Ratio', 'Batting Average']
    for i in range(len(results)):
        print('%30s | %8.3f' % (labels[i], results[i]))


def backtest(model_list, model_name_list,model,cols):

  # model1,
  for model, model_name in zip(model_list, model_name_list):
    print(model_name)
    #予測値を作成　一度やったら飛ばす モデルを変えたらもう一度やる
    prediction_csv(model,model_name,cols)

    #いくつの銘柄を買うのか
    pickup = 5


    df = pd.read_csv('/content/drive/MyDrive/machine_learning/light_gbm/backtest/prediction_{}.csv'.format(model_name), index_col=0)
    print(df.tail)
    # today_prediction = df.iloc[-1]
    today_prediction = df.iloc[-2]

    today_prediction = today_prediction.sort_values(ascending=False) 
    print(today_prediction)
    code_list = today_prediction.index.values.tolist()
    point_list = today_prediction.values.tolist()

    print(code_list[0:10])
    print(point_list[0:10])



    # df = df['2020-01-02 00:00:00':'2020-12-31 00:00:00']
    df = df.iloc[-100:-5]
    print(df)
    # df = df.drop('VAR.csv', axis=1)
    percent = []
    percentChange = []
    sell_prices =[]
    df.dropna(how='all')

    for i in range(0,len(df),5):
      series = df.iloc[i]
      series = series.sort_values(ascending=False) 
      code_list = series.index.values.tolist()
      point_list = series.values.tolist()
      date = df.index[i]
      print(date)
      print(code_list[0:pickup])
      print(point_list[0:pickup])


      for k in code_list[0:pickup]:
        data = yf.download(k.replace('.csv', ''), start="2020-01-01")
        shift_data1 = data.shift(-1)
        shift_data2 = data.shift(-6)

        # print(data[0:20])
        # print(shift_data1[0:20])
        # print(shift_data2[0:20])

        buy_price = shift_data1['Open'][date]
        sell_price = shift_data2['Open'][date]
      
        print('買値は'+str(buy_price))
        print('売値は'+str(sell_price))

        # stock_return = 100*(shift_data['Adj Close'][date]/data['Adj Close'][date]-1)
        stock_return = 100*(shift_data2['Open'][date]/shift_data1['Open'][date]-1)


        #損切りを付け加える
        if  stock_return < -15:
          stock_return = -16

        percent.append(stock_return)
      print('今回のリターン')
      print(percent)
      results = result1(percent)
      show_results(results)
      percent.clear()
      percentChange.append(results[0])

    print(percentChange)

    # print(percentChange)
    index = ['Total return', 'Average Gain', 'Average Loss', 'Max return' ,'Max Loss','Risk Reward Ratio', 'batting_average']
    results = result1(percentChange)
    print(results)
    show_results(results)

    print('========================')
    #     print(total_result.mean(axis='columns'))

    total_return = 1
    for m in percentChange:
        total_return = total_return * (1+m/100)

    print('最終リターンは'+ str((total_return - 1)*100))






# cols =['return','sma210'

#           ,'ratio_sma75_30', 'ratio_sma75_150', 'ratio_sma105_30', 
#           'ratio_sma105_150', 'ratio_sma135_150'
#           ,'Highest161','adosc', 'sma10', 
#         'adosc-SG' ,'typical-price',"ema270"
#       ,'Highest121','Highest45,81days_ago',

# 'today_by_sma30ratio','today_by_sma45ratio','today_by_sma60ratio','today_by_sma120ratio'
# ,'Adjclose_today_20daysago_ratio','Adjclose_today_15daysago_ratio','Adjclose_today_10daysago_ratio','Adjclose_today_5daysago_ratio',
#           ]


# # load model from file
# model_name1 = 'XGBboost2'
# model1 = joblib.load("/content/drive/MyDrive/machine_learning/XGBoost/model1/{}".format(model_name1))

# model_name2 = 'lightgbm_0706'
# model2 = joblib.load("/content/drive/MyDrive/machine_learning/light_gbm/model/{}.joblib".format(model_name2))

# model_name_list = [model_name2]
# model_list = [model2]

# backtest(model_list, model_name_list,model,cols)
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpl
import os



path = os.getcwd()
print(path)

data = pd.read_csv('./sAndp500.csv')
print(data)




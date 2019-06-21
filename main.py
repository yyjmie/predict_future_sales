import numpy as np
import pandas as pd
import random as rd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller,acf,pacf,arma_order_select_ic
# import statsmodels.formula.api as smf
# import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

# sales = pd.read_csv('sales_train.csv')
# print(sales.head(10))

# sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))

# print(sales.info())
# print(sales.describe())

# monthly_sales = sales.groupby(['date_block_num', 'shop_id', 'item_id'])['date', 'item_price', 'item_cnt_day'].agg({'date': 'count', 'item_price': 'mean', 'item_cnt_day': 'sum'})
# print(monthly_sales.head(10))

# ts = sales.groupby(["date_block_num"])["item_cnt_day"].sum()
# ts.astype('float')
# ts.to_csv('ts.csv')

ts = pd.read_csv('ts.csv', index_col=0)

'''
plt.figure(figsize=(16,8))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)

plt.show()

plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(), label='Rolling Mean')
plt.plot(ts.rolling(window=12,center=False).std(), label='Rolling sd')
plt.legend()

plt.show()
'''

# res = sm.tsa.seasonal_decompose(ts.values, freq=12, model='multiplicative')
res = sm.tsa.seasonal_decompose(ts.values, freq=12, model='additive')
plt.figure(figsize=(16,12))
fig = res.plot()
plt.show()







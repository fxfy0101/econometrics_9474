# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.base.datetools import dates_from_str
from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
# %%
oil = pd.read_csv(r"~/Documents/Data/oil.csv", names=["price"])
### time index
date = pd.Series(np.repeat(np.arange(1986, 2020), 12)).astype(str) + "M" + pd.Series(np.tile(np.arange(1, 13), 34)).astype(str)
date = pd.concat([date, pd.Series([2020] * 8).astype(str) + "M" + pd.Series(np.arange(1, 9)).astype(str)], ignore_index=True)
oil.index = pd.DatetimeIndex(dates_from_str(date))
# %%
### part (a) plot the detrended series in a well-labeled graph
detrend_oil = detrend(oil, order=1)
plt.plot(detrend_oil.price)
plt.xlabel("time")
plt.ylabel("detrended oil price")
plt.savefig(r"/Users/yong/Documents/Picture/oil.png", dpi=300)
# %%
### part (b) adf t-test
### linearly detrended series
print(ADF(detrend_oil.price, lags=5).summary().as_latex())
# %%
### first difference of linearly detrended series
print(ADF(detrend_oil.price.diff()[1:], lags=5).summary().as_latex())
# %%
### part (c) Phillips-Perron t-test
### linearly detrended series
print(PhillipsPerron(detrend_oil.price, lags=1).summary().as_latex())
# %%
### first difference of linearly detrended series
print(PhillipsPerron(detrend_oil.price.diff()[1:], lags=1).summary().as_latex())
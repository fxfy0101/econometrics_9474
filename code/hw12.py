# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.tsatools import detrend
from arch.unitroot import engle_granger
from arch.unitroot.cointegration import phillips_ouliaris
# %%
oilbdi = pd.read_csv(r"/Users/yong/Documents/Data/oilbdi.csv", names=["ROIL", "RBDI"])
oilbdi["detrended_ROIL"] = detrend(oilbdi["ROIL"], order=1)
oilbdi["detrended_RBDI"] = detrend(oilbdi["RBDI"], order=1)
# %%
a_results = engle_granger(oilbdi["detrended_RBDI"], oilbdi["detrended_ROIL"], trend="n", lags=5, method="t-stat")
a_results.summary()
# %%
b_results = phillips_ouliaris(oilbdi["detrended_RBDI"], oilbdi["detrended_ROIL"], trend="n", test_type="Zt")
b_results.summary()
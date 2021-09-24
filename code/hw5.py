# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stargazer.stargazer import Stargazer
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import *
# %%
### import dataset
consumptions = pd.read_csv(r"/Users/yong/Documents/Data/PCECC96.csv", names=["expenditure"])
# %%
### detrend the series
consumptions["t"] = np.arange(1, len(consumptions) + 1)
consumptions["log_expen"] = np.log(consumptions.expenditure)
reg_expen = smf.ols("log_expen ~ t", data=consumptions).fit()
detrend_expen = reg_expen.resid

def fpe(model, lag, sample_size):
    """
    statsmodels module in Python calculate fpe automatically
    but with a different method. They use the degree of freedom instead
    of original sample size to calculate sigma2. They also subtract the
    lags from the original sample size as the new sample size to
    calculate fpe.
    This function is defined according to formula in the note.
    """
    sigma2 = np.sum((model.resid)**2) / sample_size
    return sigma2 * (sample_size + lag) / (sample_size - lag)

# %%
### part (a)

### AR(4)
detrend_expen_ar_4 = AutoReg(detrend_expen, lags=4, trend="n").fit()
### FPE(4)
print(detrend_expen_ar_4.fpe)
### export the result
print(detrend_expen_ar_4.summary().as_latex())

### AR(8)
detrend_expen_ar_8 = AutoReg(detrend_expen, lags=8, trend="n").fit()
### FPE(8)
print(detrend_expen_ar_8.fpe)
### export the result
print(detrend_expen_ar_8.summary().as_latex())

### AR(12)
detrend_expen_ar_12 = AutoReg(detrend_expen, lags=12, trend="n").fit()
### FPE(12)
print(detrend_expen_ar_12.fpe)
### export the result
print(detrend_expen_ar_12.summary().as_latex())
# %%
### generate first difference series
detrend_expen_diff = detrend_expen.diff().drop(0)
# %%
### AR(4)
detrend_expen_diff_ar_4 = AutoReg(detrend_expen_diff, lags=4, trend="n").fit()
### FPE(4)
print(detrend_expen_diff_ar_4.fpe)
### export the result
print(detrend_expen_diff_ar_4.summary().as_latex())

### AR(8)
detrend_expen_diff_ar_8 = AutoReg(detrend_expen_diff, lags=8, trend="n").fit()
### FPE(8)
print(detrend_expen_diff_ar_8.fpe)
### export the result
print(detrend_expen_diff_ar_8.summary().as_latex())

### AR(12)
detrend_expen_diff_ar_12 = AutoReg(detrend_expen_diff, lags=12, trend="n").fit()
### FPE(12)
print(detrend_expen_diff_ar_12.fpe)
### export the result
print(detrend_expen_diff_ar_12.summary().as_latex())

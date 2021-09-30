# %%
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
# %%
### define HQ function
def hq(model, p, q):
    """
    llf: log likelihood function
    """
    return -2 * model.llf + 2 * (p + q) * np.log(np.log(model.nobs))
### simulate ARMA(2, 2) process
np.random.seed(1)
sample_series = arma_generate_sample([1, -2/3, 1/9], [1, 1/2, 1/2], 600)
# %%
### model estimation
### ARMA(2, 2)
arma_2_2 = ARIMA(sample_series, order=(2, 0, 2)).fit()
print(arma_2_2.summary().as_latex())
# %%
### part b
### ARMA(2, 1)
arma_2_1 = ARIMA(sample_series, order=(2, 0, 1)).fit()
print(arma_2_1.summary().as_latex())
# %%
### part b
### ARMA(2, 3)
arma_2_3 = ARIMA(sample_series, order=(2, 0, 3)).fit()
print(arma_2_3.summary().as_latex())

# %%
print(hq(arma_2_1, 2, 1))
print(hq(arma_2_2, 2, 2))
print(hq(arma_2_3, 2, 3))
# %%
### repition for different random seeds
for i in range(1, 10):
    np.random.seed(i)
    sample_series = arma_generate_sample([1, -2/3, 1/9], [1, 1/2, 1/2], 600)
    arma_2_2 = ARIMA(sample_series, order=(2, 0, 2)).fit()
    print(f"Estimation with random seed {i}:\n\t{arma_2_2.polynomial_ar}\n\t{arma_2_2.polynomial_ma}")
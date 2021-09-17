# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import acf
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.tsatools import detrend
# %%
# Problem 1 part (c)
## create lambda series
Lambda = (np.arange(0, 100) / 99) * np.pi
two_Lambda = 2 * Lambda
## calculate f(lambda) with ptf
ptf = np.abs(1 + (1/6)*np.exp(-1j*Lambda) - (1/6)*np.exp(-1j*two_Lambda)) ** (-2)
f_lambda = ptf * 1 / (2 * np.pi)
## plot
plt.figure()
plt.plot(Lambda, f_lambda, label=r"$f(\lambda)$")
plt.legend()
plt.savefig(r"/Users/yong/Documents/LaTeX/Time Series/lambda.png", dpi=300)

# %%
# Problem 1 part (d)
## simulate AR(2) process
np.random.seed(10101)
ar_2 = arma_generate_sample([1, 1/6, -1/6], [1], 1000)
ar_2_autocovariance = acf(ar_2, nlags=10, adjusted=True)
k = np.arange(0, 11)

def w_bartlett(z):
    z[np.abs(z) <= 1] = 1 - np.abs(z)
    z[np.abs(z) > 1] = 0
    return z

def f_lambda_hat(autocovariance, Lambda, window):
    return (1/(2*np.pi)) * np.sum(w_bartlett(window/window[-1]) * autocovariance * np.cos(window*Lambda))

density_estimate = [f_lambda_hat(ar_2_autocovariance, item, k) for item in Lambda]

plt.figure()
plt.plot(Lambda, f_lambda, label=r"$f(\lambda)$")
plt.plot(Lambda, density_estimate, label=r"$\hat{f}(\lambda)$")
plt.legend()
plt.savefig(r"/Users/yong/Documents/laTeX/Time Series/density_true_estimate.png", dpi=300)
# %%
# Problem 2
sales = pd.read_csv(r"/Users/yong/Documents/Data/sales.csv", names=["sales_unadj", "sales_adj"])
## linearly detrend the log of each of these series
sales_unadj_detrend = detrend(np.log(sales.sales_unadj), order=1)
sales_adj_detrend = detrend(np.log(sales.sales_adj), order=1)
### first difference of linearly detrend data
sales_unadj_detrend_diff = sales_unadj_detrend.diff().drop(0)
sales_adj_detrend_diff = sales_adj_detrend.diff().drop(0)
## set Bartlett window
k_1 = np.arange(0, 19)

# %%
## unadjusted data density
sales_unadj_density = [f_lambda_hat(acf(sales_unadj_detrend, nlags=18, adjusted=True), item, k_1) for item in Lambda]
sales_unadj_diff_density = [f_lambda_hat(acf(sales_unadj_detrend_diff, nlags=18, adjusted=True), item, k_1) for item in Lambda]
## adjusted data density
sales_adj_density = [f_lambda_hat(acf(sales_adj_detrend, nlags=18, adjusted=True), item, k_1) for item in Lambda]
sales_adj_diff_density = [f_lambda_hat(acf(sales_adj_detrend_diff, nlags=18, adjusted=True), item, k_1) for item in Lambda]
# %%
plt.plot(Lambda, sales_unadj_density, "-", Lambda, sales_unadj_diff_density, "r-")
# %%
plt.plot(Lambda, sales_adj_density, "-", Lambda, sales_adj_diff_density, "r-")
# %%
fig, ax = plt.subplots(2, 3, sharex=False, sharey=False)
fig.set_figwidth(18)
fig.set_figheight(10)
ax[0, 0].plot(sales_unadj_detrend)
ax[0, 0].set_title("detrend log unadjusted sales")
ax[0, 1].plot(sales_unadj_detrend_diff)
ax[0, 1].set_title("detrend log unadjusted sales diff")
ax[0, 2].plot(Lambda, sales_unadj_density, label="log series")
ax[0, 2].plot(Lambda, sales_unadj_diff_density, label="first difference")
ax[0, 2].legend()
ax[0, 2].set_title("densities")
ax[1, 0].plot(sales_adj_detrend)
ax[1, 0].set_title("detrend log adjusted sales")
ax[1, 1].plot(sales_adj_detrend_diff)
ax[1, 1].set_title("detrend log adjusted sales diff")
ax[1, 2].plot(Lambda, sales_adj_density, label="log series")
ax[1, 2].plot(Lambda, sales_adj_diff_density, label="first difference")
ax[1, 2].legend()
ax[1, 2].set_title("densities")
plt.savefig(r"/Users/yong/Documents/LaTeX/Time Series/densities.png", dpi=300)
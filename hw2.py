import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import acf


gamma = [152/27, 535/108, 205/54]
for _ in range(7):
    gamma.append(gamma[-1] - (1/4)*gamma[-2])
### calculate autocovariance
rho_yulewalker = np.array(gamma) / gamma[0]

### simulate ARMA
### X_t = X_{t - 1} - 1/4(X_{t - 2}) + e_t + 1/4e_{t - 1} + 1/4e_{t - 2}
np.random.seed(10101)
epsilon = np.random.randn(1000000)
### set initial values
x_0 = 0.5
x_1 = 0.5 + epsilon[0]
x = [x_0, x_1]
### generate ARMA(2, 2) series
for i in range(10000):
    (x.append(x[-1] - (1/4)*x[-2] + epsilon[i + 2] + 
    (1/4)*epsilon[i + 1] + (1/4)*epsilon[i]))
### calculate autocovariance
rho_sim = acf(x, nlags=9)

plt.figure()
plt.plot(rho_yulewalker, "-.", label="Yule Walker Method")
plt.plot(rho_sim, "-.", label="Nonparametric Estimation")
plt.xlabel("lag")
plt.ylabel(r"$\rho(k)$")
plt.legend()
plt.savefig(r"/Users/yong/Documents/LaTeX/Time Series/rho_k.png", dpi=300)

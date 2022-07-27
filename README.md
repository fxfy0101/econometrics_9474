# 9474 Advanced Topics in Econometrics I

## Coding Question Solutions

### Homework 1

- Use data in the file `anomaly.csv`. Plot temperature anomalies over time.
- Estimate and plot the sample autocorrelation functions of the raw series and its first difference.

### Homework 2

- Given an ARMA(2, 2) process with alpha lag polynominal [1, -1, 1/4] and beta lag polynominal [1, 1/4, 1/4], graph the autocorrelation up to k = 9. Use Yule-Walker and MA technique to calculate autocovariance.

### Homework 3

- Given an AR(2) process with alpha lag polynominal [1, 1/6, -1/6], use the power transfer function to plot the densities over [0, pi].
- Simulate an AR(2) process of length $n = 1000$ and then estimate the densities with a Bartlett window of 10.
- Use data in the file `sales.csv`. Linearly detrend the log of each series. Estimate the spectral densities of the two series and their first differences using a Bartlett window of 18.

### Homework 5

- Use data in the file `PCECC96.csv`. Detrend the series by regressing the natural log onto a constant and a linear time trend. The detrended series is the residual series from this regression.
- Fit an AR(p) to the detrended series using FPE(p) with p = [4, 8, 12].
- Fit an AR(p) to the first-differenced detrended series using FPE(p) [4, 8, 12].

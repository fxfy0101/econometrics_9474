library(tidyverse)

### import the dataset
anomaly <- read_csv("Documents/Data/anomaly.csv", col_names = FALSE)

### change variable name and generate first order difference
anomaly <- rename(anomaly, temp = X1)
### set time series
temp <- ts(anomaly$temp, start = 1850, frequency = 1)

### plot the temperature anomalies
png(filename = "Documents/Data/temp_anomalies.png", width = 8, height = 6, 
    units = "in", res = 300)
plot(temp, xlab = "time", ylab = "temperature", family = "serif")
dev.off()

### calculate the sample autocorrelation function
acf(temp, lag.max = 9, plot = FALSE)
### plot the sample autocorrelation functions
png(filename = "Documents/Data/raw_acf.png", width = 8, height = 6, 
    units = "in", res = 300)
acf(temp, lag.max = 9, xlab = "lags", main = "", family = "serif")
dev.off()

### calculate the sample autocorrelation function
acf(diff(temp), lag.max = 9, plot = FALSE)
png(filename = "Documents/Data/diff_acf.png", width = 8, height = 6, 
    units = "in", res = 300)
acf(diff(temp), lag.max = 9, xlab = "lags", main = "", family = "serif")
dev.off()

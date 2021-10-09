# %%
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
# %%
### import dataset
gdp = pd.read_csv(r"/Users/yong/Documents/Data/GDPC96.csv", names=["realgdp"])
wti = pd.read_csv(r"/Users/yong/Documents/Data/OILPRICE.csv", names=["WTI"])
ppi = pd.read_csv(r"/Users/yong/Documents/Data/PPIACO.csv", names=["PPI"])
# %%
### create time index for each series
date = pd.Series(np.repeat(np.arange(1950, 2014), 4)).astype(str) + "Q" + pd.Series(np.tile(np.arange(1, 5), 64)).astype(str)
gdp.index = pd.DatetimeIndex(dates_from_str(date))
ppi.index = pd.DatetimeIndex(dates_from_str(date))
wti.index = pd.DatetimeIndex(dates_from_str(np.repeat(date, 3)))
# %%
### aggregate monthly wti to quarterly average
wti = wti.groupby(level=0).mean()
### merge datasets
oildata = gdp.merge(wti, right_index=True, left_index=True).merge(ppi, left_index=True, right_index=True)
oildata["realoilprice"] = oildata.WTI / oildata.PPI
### calculate the first difference of log series
oildata = np.log(oildata).diff().dropna()
# %%
### bivariate VAR(4)
model_a = VAR(oildata[["realgdp", "realoilprice"]])
results_a = model_a.fit(maxlags=4)
### export result
estimates = pd.DataFrame({"coeff_gdp": results_a.params.realgdp.values, "se_gdp": results_a.bse.realgdp.values, 
                          "coeff_oil": results_a.params.realoilprice.values, "se_oil": results_a.bse.realoilprice.values})
estimates.index = pd.Index(results_a.params.index.values)
estimates.columns = pd.MultiIndex.from_arrays([["Equation: Real GDP", "Equation: Real GDP", "Equation: Real Oil Price", "Equation: Real Oil Price"], 
                                               ["Coefficient", "Standard Error", "Coefficient", "Standard Error"]])
print(estimates.to_latex())
# %%
### granger causality
print(results_a.test_causality("realgdp", "realoilprice").summary().as_latex_tabular())
print(results_a.test_causality("realoilprice", "realgdp").summary().as_latex_tabular())
# %%
### create positive and negative series
oildata["pos_realoilprice"] = oildata["realoilprice"].copy()
oildata["pos_realoilprice"][lambda x: x < 0] = 0
oildata["neg_realoilprice"] = oildata["realoilprice"].copy()
oildata["neg_realoilprice"][lambda x: x > 0] = 0
# %%
### VAR(4) with r = 3
model_d = VAR(oildata[["realgdp", "pos_realoilprice", "neg_realoilprice"]])
results_d = model_d.fit(maxlags=4)
# %%
### export result
params = results_d.params
estimates_d = pd.DataFrame({"Coefficient": params.realgdp.values, "Standard Error": results_d.bse.realgdp.values})
estimates_d.index = results_d.bse.index
estimates_d.columns = pd.MultiIndex.from_arrays([["Equation: Real GDP Growth", "Equation: Real GDP Growth"], ["Coefficient", "Standard Error"]])
print(estimates_d.to_latex())
# %%
### obtain the variance covariance matrix of coefficient estimates
vcov = results_d.cov_params()
### perform hypothesis test
def t_test(param_1, param_2):
    test_stattistic = (params["realgdp"][param_1] - params["realgdp"][param_2]) / (np.sqrt(vcov[param_1]["realgdp"][param_1]["realgdp"] + vcov[param_2]["realgdp"][param_2]["realgdp"] - 2*vcov[param_1]["realgdp"][param_2]["realgdp"]))
    return test_stattistic
test_result = []
lag_coef = []
for i in range(1, 5):
    test_result.append(t_test(f"L{i}.pos_realoilprice", f"L{i}.neg_realoilprice"))
    lag_coef.append(f"Lag {i} Coefficient")
### export result
test_result_df = pd.DataFrame({"Test Statistic": test_result}, index=lag_coef)
print(test_result_df.to_latex())
# %%

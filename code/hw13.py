# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.tsatools import detrend
# %%
oilbdi = pd.read_csv(r"/Users/yong/Downloads/oilbdi.csv", names=["ROIL", "RBDI"])
oilbdi_detrend = detrend(oilbdi)
# %%
def johansen_results(res):
    df = pd.DataFrame({"trace statistics": res.trace_stat, "90%": res.trace_stat_crit_vals[:, 0], "95%": res.trace_stat_crit_vals[:, 1], "99%": res.trace_stat_crit_vals[:, 2]})
    df.columns = [["", "Critical Values", "Critical Values", "Critical Values"], ["trace statistics", "90%", "95%", "99%"]]
    return df
# %%
### p = 0
johansen_results(coint_johansen(oilbdi_detrend, -1, 0))
# %%
### p = 1
johansen_results(coint_johansen(oilbdi_detrend, -1, 1))
# %%
### p = 2
johansen_results(coint_johansen(oilbdi_detrend, -1, 2))
# %%
### p = 3
johansen_results(coint_johansen(oilbdi_detrend, -1, 3))
# %%
### p = 4
johansen_results(coint_johansen(oilbdi_detrend, -1, 4))
# %%
### p = 5
print(johansen_results(coint_johansen(oilbdi_detrend, -1, 5)).to_latex())
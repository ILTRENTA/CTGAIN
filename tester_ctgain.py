import torch 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from ctgain.ctgain import CTGAIN
from ctgain.utils.ctgain import *
from ctgain.utils.GAIN_metrics import *

plt.style.use("ggplot")
wine=load_wine()

data=pd.DataFrame(wine.data, columns=wine.feature_names)
data.head()

data=pd.read_csv("datasets/clean_datasets/abalone.csv")

cols=data.columns


trans_dta=produce_NA(data.values, p_miss=.2, p_obs=.2, mecha="MAR")


trans_dta=produce_NA(data.values, p_miss=.4, mecha="MCAR")



incomp_dta=trans_dta["X_incomp"]


incomp_dta=pd.DataFrame(incomp_dta, columns=cols)
incomp_dta.head()


model=CTGAIN(epochs=200, verbose=True,
              batch_size=256,
             hint_rate=0.6, alpha=1)

model.fit(incomp_dta)
print(pd.DataFrame(trans_dta["X_init"],columns=cols))
print(incomp_dta)

mask=1-incomp_dta.isnull().astype(int).values

# imputed, _, _, _=model.impute(incomp_dta, real_only=False)

imputed=model.impute(incomp_dta.copy())
print(imputed)

print(rmse_loss(data.values,imputed.values ,mask))

plt.scatter(data["LongestShell"], imputed["LongestShell"], label="original")
plt.show()
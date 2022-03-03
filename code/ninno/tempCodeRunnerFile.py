#BEFORE RUNNING THE CODE, RUN THE FOLLOWING COMMANDS ON THE TERMINAL
'''
pip install pandas
pip install statsmodels
pip install linearmodels
#'''


import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.sandbox.regression import gmm
import os
os.getcwd()
import shutil
from pathlib import Path
sep=";"
current_dir="/Users/rubenexojo/Library/Mobile Documents/com~apple~CloudDocs/phd/uni/courses/finance/finance-3/hw/code/ninno/"
#SUMMARY COL
def sum_col(n, cols, x, lags, regressors):
    regs=[None]*len(cols)
    for i, col in enumerate(cols):
        y=df[col]
        reg=sm.OLS(y, x).fit(cov_type='HAC',cov_kwds={'maxlags':lags})
        regs[i]=reg
    new_cols=[i.replace("A", "").replace("ex_", "").replace("R", "") for i in cols]
    sum=summary_col(regs, float_format='%0.2f', model_names=(new_cols), stars=True, info_dict=None, regressor_order=(regressors), drop_omitted=True)
    text=sum.as_latex()
    p=Path(f"tex/row_{n}.txt")
    p.write_text(f"{text}\n\n")


#############################################################################
#TABLE 1
#############################################################################
regressors=["constant"]
file_path="FF4_monthlyKF.csv"
df=pd.read_csv(file_path, sep=sep, names=["dtt", "v2", "v3", "v4", "rf"], dtype=str)
df=df[["dtt", "rf",]]
df2=df.copy()
file_path="Data_master.xlsx"
df=pd.read_excel(file_path, sheet_name="T1_ptfs", names=["dt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
cols=["LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong"]
for i, col in enumerate(cols):
    new_col=f"A{col}"
    df[new_col]=pd.to_numeric(df[col], errors='coerce')*12*100
df["dtt"]=df["dt"].str.slice(0, 6)
cols=["mktrf", "smb", "hml"]
for i, col in enumerate(cols):
    df[col]=pd.to_numeric(df[col], errors='coerce')
cols=["LL", "LLStrong"]
for i, col in enumerate(cols):
    new_col=f"ex_{col}"
    df[new_col]=pd.to_numeric(df[col], errors='coerce')*12*100
df1=df.copy()
df=df1.merge(df2, how='inner', on="dtt", left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_left', '_right'), copy=True, indicator=True, validate="one_to_one")
df_merged=df.copy()
cols=["Lead", "Mid", "Lag"]
for i, col in enumerate(cols):
    new_col=f"ex_{col}"
    df[new_col]=pd.to_numeric(df[col], errors='coerce')*12*100-pd.to_numeric(df["rf"], errors='coerce')*12

#TABLES
shutil.rmtree('tex/', ignore_errors=True)
p=Path(f"{current_dir}/tex")
p.mkdir(mode=511, parents=False, exist_ok=True)

df["constant"]=[1]*len(df)
df["dt"]=pd.to_datetime(df["dt"], format="%Y%m%d")
df=df.set_index(["dt"]).sort_index()
text=""

#ROW 1
n=1
cols=["ALeadR", "AMidR", "ALagR", "ALL", "ALLStrong"]
x=df["constant"]
lags=24
print(x)
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("constant", "Average return")
text=f"{text}\n{txt}"
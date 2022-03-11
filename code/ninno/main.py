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

#ROW 2
n=2
cols=["ex_Lead", "ex_Mid", "ex_Lag", "ex_LL", "ex_LLStrong"]
x=df[["constant", "mktrf"]]
lags=24
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("constant", "CAPM alpha")
text=f"{text}\n{txt}"

#ROW 3
n=3
cols=["ex_Lead", "ex_Mid", "ex_Lag", "ex_LL", "ex_LLStrong"]
x=df[["constant", "mktrf", "smb", "hml"]]
lags=24
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("constant", "FF3 alpha")

text=f"{text}\n{txt}"
title="{Lead-Lag Portfolio Sorting (Max Correlation)}"
text=text.replace("\caption{}", f"\caption{title}")
j=text.find("R-squared")
k=text.find("CAPM alpha", j)-len("\hline")-1
start=text[:j]
end=text[k:]
text=f"{start}{end}"
j=text.find("R-squared")
k=text.find("FF3 alpha", j)-len("\hline")-1
start=text[:j]
end=text[k:]
text=f"{start}{end}"
j=text.find("R-squared")
k=text.find("\end{tabular}", j)-len("\hline")-1
start=text[:j]
end=text[k:]
text=f"{start}{end}"

#############################################################################
#TABLE 3
#############################################################################
regressors=["constant"]
file_path="Data_master.xlsx"
df=pd.read_excel(file_path, sheet_name="T3_ptfs", names=["dt", "LL38", "LLStrong38", "LL49", "LLStrong49"], dtype=str)
cols=["LL38", "LLStrong38", "LL49", "LLStrong49"]
for i, col in enumerate(cols):
    new_col=f"A{col}"
    df[new_col]=pd.to_numeric(df[col], errors='coerce')*12*100
df["dtt"]=df["dt"].str.slice(0, 6)
df3=df.copy()

#TABLES
df["constant"]=[1]*len(df)
df["dt"]=pd.to_datetime(df["dt"], format="%Y%m%d")
df=df.set_index(["dt"]).sort_index()

#ROW 4
n=4
cols=["ALL38", "ALLStrong38", "ALL49", "ALLStrong49"]
x=df["constant"]
lags=12
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("constant", "Average Return")
text=f"{text}\n{txt}"

df=df3.merge(df1, how='inner', on="dt", left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_left', '_right'), copy=True, indicator=True, validate="one_to_one")

df["constant"]=[1]*len(df)
df["dt"]=pd.to_datetime(df["dt"], format="%Y%m%d")
df=df.set_index(["dt"]).sort_index()

#ROW 5
n=5
cols=["ALL38", "ALLStrong38", "ALL49", "ALLStrong49"]
x=df[["constant", "mktrf"]]
lags=12
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("constant", "CAPM alpha")
text=f"{text}\n{txt}"

#ROW 6
n=6
cols=["ALL38", "ALLStrong38", "ALL49", "ALLStrong49"]
x=df[["constant", "mktrf", "smb", "hml"]]
lags=12
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("constant", "FF3 alpha")

text=f"{text}\n{txt}"
title="{The Disconnect between LL and Other Factors (II)}"
text=text.replace("\caption{}", f"\caption{title}")
j=text.find("R-squared")
k=text.find("CAPM alpha", j)-len("\hline")-1
start=text[:j]
end=text[k:]
text=f"{start}{end}"
j=text.find("R-squared")
k=text.find("FF3 alpha", j)-len("\hline")-1
start=text[:j]
end=text[k:]
text=f"{start}{end}"
j=text.find("R-squared")
k=text.find("\end{tabular}", j)-len("\hline")-1
start=text[:j]
end=text[k:]
text=f"{start}{end}"

#############################################################################
#TABLE 7
#############################################################################
file_path="Data_master.xlsx"
df=pd.read_excel(file_path, sheet_name="T7_factors", names=["dt", "LLStrong30", "mktrf", "smb", "hml", "rmw", "cma", "q_mkt", "q_me", "q_ia", "q_roe", "L", "M", "N", "mom"], dtype=str)
cols=["LLStrong30", "mktrf", "smb", "hml", "rmw", "cma", "q_mkt", "q_me", "q_ia", "q_roe", "L", "M", "N", "mom"]
for i, col in enumerate(cols):
    new_col=f"A{col}"
    df[new_col]=pd.to_numeric(df[col], errors='coerce')*12*100
df["dtt"]=df["dt"].str.slice(0, 6)

#TABLES
df["constant"]=[1]*len(df)
df["dt"]=pd.to_datetime(df["dt"], format="%Y%m%d")
df=df.set_index(["dt"]).sort_index()

#ROW
n=8
cols=["ALLStrong30"]
x=df[["Amktrf", "Asmb", "Ahml", "Armw", "Acma"]]
lags=12
regressors=list(x.columns)
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("LLStrong30", "FF5").replace("constant", "alpha LL").replace("Amktrf", "MKT").replace("Asmb", "SMB").replace("Ahml", "HML").replace("Armw", "RMW").replace("Acma", "CMA")
text=f"{text}\n{txt}"

#ROW 9
n=9
cols=["ALLStrong30"]
x=df[["constant", "Aq_mkt", "Aq_me", "Ahml", "Aq_ia", "Aq_roe"]]
lags=12
regressors=list(x.columns)
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("LLStrong30", "HXZ q-factors").replace("constant", "alpha LL").replace("Aq\_mkt", "MKT").replace("Aq\_me", "ME").replace("Ahml", "HML").replace("Aq\_ia", "I/A").replace("Aq\_roe", "ROE")
text=f"{text}\n{txt}"

#ROW 10
n=10
y=df[["ALLStrong30"]]

x=df[["constant", "Amom"]]
lags=12
reg10=sm.OLS(y, x).fit(cov_type='HAC',cov_kwds={'maxlags':lags})

x=df[["constant", "AL", "Amom"]]
reg11=sm.OLS(y, x).fit(cov_type='HAC',cov_kwds={'maxlags':lags})

x=df[["constant", "AL", "Amom", "AM", "AN"]]
reg12=sm.OLS(y, x).fit(cov_type='HAC',cov_kwds={'maxlags':lags})

regs=[reg10, reg11, reg12]
sum=summary_col(regs, float_format='%0.2f', model_names=(["Carhart MOM", "Carhart MOM", "Carhart MOM"]), stars=True, info_dict=None, regressor_order=(["constant", "AL", "Amom", "AM", "AN"]), drop_omitted=True)
txt=sum.as_latex()
p=Path(f"tex/row_{n}.txt")
p.write_text(f"{txt}\n\n")
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("constant", "alpha LL").replace("AL", "MKT").replace("Amom", "MOM").replace("AM", "SMB").replace("AN", "HML")
text=f"{text}\n{txt}"

#############################################################################
#TABLE 9
#############################################################################
file_path="30_industry_pfs.csv"
n=int(file_path[:2])
inds=[None]*n
for i in range(n):
    inds[i]=f"Ind{i+1}"
vars=["dtt"]+inds
df=pd.read_csv(file_path, sep=sep, names=vars, dtype=str)
df4=df.copy()
df=df4.merge(df_merged, how='inner', on="dtt", left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_left', '_right'), copy=True, indicator=False, validate="one_to_one")

cols=inds
for i, col in enumerate(cols):
    new_col=f"ex_{col}"
    df[new_col]=pd.to_numeric(df[col], errors='coerce')-pd.to_numeric(df["rf"], errors='coerce')
df["LL"]=pd.to_numeric(df["LL"], errors='coerce')*100

#TABLES
df["constant"]=[1]*len(df)
df["dtt"]=pd.to_datetime(df["dtt"], format="%Y%m")
df[(df["dtt"]>="1972-01") & (df["dtt"]<="2012-12")]
df=df.set_index(["dtt"]).sort_index()

#https://bashtage.github.io/linearmodels/
#https://bashtage.github.io/linearmodels/asset-pricing/asset-pricing/linearmodels.asset_pricing.model.LinearFactorModelGMM.html#linearmodels.asset_pricing.model.LinearFactorModelGMM
#https://bashtage.github.io/linearmodels/system/examples/three-stage-ls.html#System  -GMM-Estimation



'''#ROW 11
n=11
cols=["ALLStrong30"]
x=df[["Amktrf", "Asmb", "Ahml", "Armw", "Acma"]]
lags=12
regressors=list(x.columns)
sum_col(n, cols, x, lags, regressors)
p=Path(f"tex/row_{n}.txt")
txt=p.read_text()
txt=txt.replace("LLStrong30", "FF5").replace("constant", "alpha LL").replace("Amktrf", "MKT").replace("Asmb", "SMB").replace("Ahml", "HML").replace("Armw", "RMW").replace("Acma", "CMA")
text=f"{text}\n{txt}"
#'''




#OUTPUT
p=Path("tex/tables.txt")
p.write_text(f"{text}\n\n")
begin="""\\documentclass{report}
\\usepackage{booktabs}
\\title{Homework - Finance 3}
\\author{Nicola Maria Fiore}
\\date{March 2022}
\\begin{document}
\\maketitle
"""
end="\end{document}"
p=Path("tex/tables.txt")
text=p.read_text()


p=Path("tex/output.tex")
p.write_text(f"{begin}\n\n{text}\n\n{end}")

df.to_csv("results.csv")
print("done")

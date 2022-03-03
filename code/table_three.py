import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from pathlib import Path

class Table3(object):

    def __init__(self, master_file, rf_file):
        # DATA SETS
        self.df     = pd.read_excel(master_file, sheet_name="T3_ptfs", names=["dt", "LL38", "LLStrong38", "LL49", "LLStrong49"], dtype=str)
        self.df_rf  = pd.read_csv(rf_file, sep=";", names=["dtt", "v2", "v3", "v4", "rf"], dtype=str)
        self.df_rf  = self.df_rf[["dtt", "rf"]]
        self.df2    = self.df_rf.copy()
        self.df_ff  = pd.read_excel(master_file, sheet_name="T1_ptfs", names=["dttt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
        self.df_ff  = self.df_ff[["dttt", "mktrf", "smb", "hml"]]
        self.df3    = self.df_ff.copy()

        self.ann    = 12
        self.lags   = 24
        # Factors
        self.F      = self.df.keys().tolist()
        self.K      = len(self.F)
        self.zip_f  = {}
        self.T      = len(self.df)

    def _take_factors(self):
        self.F = self.df.keys().tolist()
        self.K = len(self.F)

    def _factors_order(self):
        flags   = [k for k in range(self.K)]
        for flag in flags:
            self.zip_f[self.F[flag]] = flag
        self.zip_f
    
    def _df_include_constant_columns(self):
        self.df["const"] = [1]*self.T
        cols    = self.df.columns.tolist()
        cols    = [cols[0]] + ["const"] + cols[1:-1]
        self.df = self.df[cols]

    def _remove_already_annualised(self, cols):
        if "dt" in cols:
            cols.remove("dt")
        if "mktrf" in cols:
            cols.remove("mktrf")
        if "smb" in cols:
            cols.remove("smb")
        if "hml" in cols:
            cols.remove("hml")
        if "const" in cols:
            cols.remove("const")
        return cols

    def _df_annualise(self):
        cols = self.df.columns.tolist()
        cols = self._remove_already_annualised(cols)
        for col in cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')*self.ann*100

    def _df_merge(self):
        self.df["dtt"] = self.df["dt"].str.slice(0, 6)
        self.df = self.df.merge(self.df2, how='inner', on="dtt", left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_left', '_right'), copy=True, indicator=True, validate="one_to_one")
        self.df.drop('dtt', inplace=True, axis=1)
        self.df.drop('_merge', inplace=True, axis=1)
        self.df["dttt"] = self.df["dt"].str.slice(0, 8)
        self.df = self.df.merge(self.df3, how='inner', on="dttt", left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_left', '_right'), copy=True, indicator=True, validate="one_to_one")
        self.df.drop('dttt', inplace=True, axis=1)
        self.df.drop('_merge', inplace=True, axis=1)
    

    def _take_factor_sheet(self): 
        # include a column of ones to calculate the average return
        self._df_include_constant_columns()
        # annualise data (we have it daily)
        self._df_annualise()
        # merge
        self._df_merge()

    def _LL_portfolio_sorting(self):
        return [self.F[self.zip_f['LL38']], self.F[self.zip_f['LLStrong38']], self.F[self.zip_f['LL49']], self.F[self.zip_f['LLStrong49']]]

    def _tex_file(self, title):
        return "tex/T3_" + title + ".txt"
        
    def _create_txt(self, title, text):
        file    = Path(self._tex_file(title))
        file.write_text(f"{text}\n\n")

    def _reg(self, Y, X, title):
        _regs=[None]*len(Y)
        for i, y in enumerate(Y):
            y   = self.df[y]
            _reg = sm.OLS(y.astype(float), X.astype(float)).fit(cov_type='HAC', cov_kwds={'maxlags':self.lags})
            _regs[i]=_reg  
        new_Y=[y.replace("ex_", "") for y in Y]
        sum = summary_col(results=_regs, float_format='%0.2f', model_names=new_Y, stars=True, regressor_order=(["const"]),info_dict=None,  drop_omitted=True)
        text    = sum.as_latex()
        self._create_txt(title, text)
    
    def _average_return(self):
        Y       = self._LL_portfolio_sorting()
        X       = self.df["const"]
        self._reg(Y, X, "average_return")

    def _capm(self):
        Y       = self._LL_portfolio_sorting()
        X       = self.df[["const", "mktrf"]]
        self._reg(Y, X, "capm")

    def _FF(self):
        Y       = self._LL_portfolio_sorting()
        X       = self.df[["const", "mktrf", "smb", "hml"]]
        self._reg(Y, X, "ffm")

    def Table3(self):
        self._take_factor_sheet()
        self._take_factors()
        self._factors_order()
        # first row (average return)
        self._average_return()
        # second row (capm)
        self._capm()
        # third row (ff3)
        self._FF()


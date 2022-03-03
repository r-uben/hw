import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from pathlib import Path

class Table1(object):

    def __init__(self, master_file, rf_file):

        self.df     = pd.read_excel(master_file, sheet_name="T1_ptfs", names=["dt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
        self.df_rf  = pd.read_csv(rf_file, sep=";", names=["dtt", "v2", "v3", "v4", "rf"], dtype=str)
        #
        self.ann    = 12
        self.lags   = 24
        # Factors
        self.F      = self.df.keys().tolist()
        self.K      = len(self.F)
        self.zip_f  = {}
        self.df_rf  = self.df_rf[["dtt", "rf"]]
        self.df2    = self.df_rf.copy()
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

    def fix_date(self, date):
        year    = [date[i] for i in range(4)]
        month   = [date[i] for i in range(4,6)]
        day     = [date[i] for i in range(6,8)]
        year    = ''.join(year)
        month   = ''.join(month)
        day     = ''.join(day)
        date = [day, month, year]
        date = '/'.join(date)
        return date

    def _df_fix_date(self):
        date_col = self.df['dt']
        for i in range(self.T):
            date_col[i] = self.fix_date(str(date_col[i]))
        self.df['dt'] = date_col
        
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

    def _df_excess_returns(self, annualised = True):
        all_cols =  self.df.columns.tolist()
        rf = pd.to_numeric(self.df["rf"], errors='coerce')*self.ann
        cols=["Lead", "Mid", "Lag"]
        move = 1
        for col in cols:
            new_col=f"ex_{col}"
            if annualised:
                loc = all_cols.index(col) + move
                self.df.insert(loc, new_col, pd.to_numeric(self.df[col], errors='coerce') - rf)
            else:
                loc = all_cols.index(col) + move
                self.df.insert(loc, new_col, pd.to_numeric(self.df[col], errors='coerce')*self.ann*100 - rf)
            move += 1
        cols=["LL", "LLStrong"]
        for col in cols:
            new_col=f"ex_{col}"
            if annualised:
                loc = all_cols.index(col) + move
                self.df.insert(loc, new_col, pd.to_numeric(self.df[col], errors='coerce'))
            else:
                self.df.insert(loc, new_col, pd.to_numeric(self.df[col], errors='coerce'))*self.ann*100
            move += 1

    def _df_merge(self):
        self.df["dtt"] = self.df["dt"].str.slice(0, 6)
        df1 = self.df.copy()
        self.df = df1.merge(self.df2, how='inner', on="dtt", left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_left', '_right'), copy=True, indicator=True, validate="one_to_one")
        self.df.drop('dtt', inplace=True, axis=1)
        self.df.drop('_merge', inplace=True, axis=1)

    def _take_factor_sheet(self): 
        # include a column of ones to calculate the average return
        self._df_include_constant_columns()
        # annualise data (we have it daily)
        self._df_annualise()
        # merge
        self._df_merge()
        # include columns of excess returns
        self._df_excess_returns()

    def _LL_portfolio_sorting(self, excess = False):
        if excess:
            return [self.F[self.zip_f['ex_Lead']], self.F[self.zip_f['ex_Mid']], self.F[self.zip_f['ex_Lag']], self.F[self.zip_f['LL']], self.F[self.zip_f['LLStrong']]]
        else:
            return [self.F[self.zip_f['LeadR']], self.F[self.zip_f['MidR']], self.F[self.zip_f['LagR']], self.F[self.zip_f['LL']], self.F[self.zip_f['LLStrong']]]

    def _tex_file(self, title):
        return "tex/T1_" + title + ".txt"
        
    def _create_txt(self, title, text):
        file    = Path(self._tex_file(title))
        file.write_text(f"{text}\n\n")

    def _reg(self, Y, X, title):
        regs=[None]*len(Y)
        for i, y in enumerate(Y):
            y   = self.df[y]
            reg = sm.OLS(y.astype(float), X.astype(float)).fit(cov_type='HAC', cov_kwds={'maxlags':self.lags})
            regs[i]= reg  
        new_Y=[y.replace("ex_", "") for y in Y]
        sum = summary_col(results= regs, float_format='%0.2f', model_names=new_Y, stars=True, regressor_order=(["const"]), info_dict=None,  drop_omitted=True)
        text    = sum.as_latex()
        self._create_txt(title, text)

    def _average_return(self):
        Y       = self._LL_portfolio_sorting()
        X       = self.df["const"]
        self._reg(Y, X, "average_return")
    
    def _capm(self):
        Y       = self._LL_portfolio_sorting(True)
        X       = self.df[["const", "mktrf"]]
        self._reg(Y, X, "capm")

    def _FF(self):
        Y       = self._LL_portfolio_sorting(True)
        X       = self.df[["const", "mktrf", "smb", "hml"]]
        self._reg(Y, X, "ffm")

    def Table1(self):
        self._take_factor_sheet()
        self._take_factors()
        self._factors_order()
        # first row (average return)
        self._average_return()
        # second row (capm)
        self._capm()
        # third row (ff3)
        self._FF()
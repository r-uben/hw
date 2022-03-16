import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from aux_class import AuxClass

class Table7(object):

    def __init__(self, master_file, rf_file, lags):
        self.aux = AuxClass(master_file, rf_file)
        # DATA SETS
        # Main data set: T7_factors
        self.header         = ["dt", "LLStrong30", "mktrf", "smb", "hml", "rmw", "cma", "q_mkt", "q_me", "q_ia", "q_roe", "mktrf2", "smb2", "hml2", "mom"]
        self.ff5            = ["const", "mktrf", "smb", "hml", "rmw", "cma"]
        self.hxz            = ["const", "q_mkt", "q_me", "q_ia", "q_roe"]
        self.carhart_mom    = ["const", "mktrf2", "smb2", "hml2", "mom"]
        self.df             = pd.read_excel(master_file, sheet_name="T7_factors", names=self.header, dtype=str)
        self.df["dt"]       = pd.to_datetime(self.df["dt"], format="%Y%m%d")
        self.df             = self.df.set_index(["dt"]).sort_index()
        # FF4 (risk-free interest rates)
        self.df_rf  = pd.read_csv(rf_file, sep=";", names=["dtt", "v2", "v3", "v4", "rf"], dtype=str)
        self.df_rf  = self.df_rf[["dtt", "rf"]]
        self.df2    = self.df_rf.copy()
        # Aux: T1_ptfs
        self.df_ff  = pd.read_excel(master_file, sheet_name="T1_ptfs", names=["dttt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
        self.df_ff  = self.df_ff[["dttt", "mktrf", "smb", "hml"]]
        self.df3    = self.df_ff.copy()
        # Annualiser factor
        self.ann    = 12
        # Number of lags
        self.lags   = lags
        # Number of actors
        self.F      = self.df.keys().tolist()
        self.K      = len(self.F)
        self.zip_f  = {}
        # Number of periods
        self.T      = len(self.df)
    
    def _df_include_constant_columns(self):
        self.df["const"] = [1]*self.T
        cols    = self.df.columns.tolist()
        cols    = ["const"] + cols[:-1]
        self.df = self.df[cols]

    def _df_remove_cols_to_not_annualise(self, cols):
        if "dt" in cols:
            cols.remove("dt")
        if "const" in cols:
            cols.remove("const")
        return cols

    def _df_annualise(self):
        cols = self.df.columns.tolist()
        cols = self._df_remove_cols_to_not_annualise(cols)
        for col in cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')*self.ann*100
    
    def _df_take_factor_sheet(self): 
        # include a column of ones to calculate the average return
        self._df_include_constant_columns()
        self._df_annualise()

    def _LL_portfolio(self):
        return [self.F[self.zip_f['LLStrong30']]]

    def _reg(self, Y, X, title):
        if type(X) is not list:
            y   = self.df[Y]

            _reg = sm.OLS(y.astype(float), X.astype(float)).fit(cov_type='HAC', cov_kwds={'maxlags':self.lags})
            sum = summary_col(results=[_reg], float_format='%0.2f', model_names=Y, stars=True,info_dict=None,  drop_omitted=True)
            text    = sum.as_latex()
            self.aux.create_txt("T7", title, text)  
        else:
            _regs=[]
            y = self.df[Y]
            for i, x in enumerate(X):
                _reg = sm.OLS(y.astype(float), x.astype(float)).fit(cov_type='HAC', cov_kwds={'maxlags':self.lags})
                _regs.append(_reg)
                sum = summary_col(results=_regs, float_format='%0.2f', model_names=Y, stars=True,info_dict=None,  drop_omitted=True)
                text    = sum.as_latex()
                self.aux.create_txt("T7_MOM" + str(i+1), title, text)
                _regs=[]

    def _FF5(self):
        Y       = self._LL_portfolio()
        X       = self.df[self.ff5]
        self._reg(Y, X, "FF5")

    def _HXZ(self):
        Y       = self._LL_portfolio()
        X       = self.df[self.hxz]
        self._reg(Y, X, "HXZ")

    def _MOM(self):
        Y            = self._LL_portfolio()
        missing_reg1 = ["mktrf2", "smb2", "hml2"]
        missing_reg2 = ["smb2", "hml2"]
        missing_reg3 = []
        missing_regs = [missing_reg1, missing_reg2, missing_reg3]
        X = []
        for missing_reg in missing_regs:
            X.append(self.df[self.aux.remove_elements(self.carhart_mom, missing_reg)])
            self.carhart_mom = self.carhart_mom + missing_reg
        self._reg(Y, X, "MOM")

    def Table7(self):
        self._df_take_factor_sheet()
        self.F, self.K  = self.aux.take_factors(self.df)
        self.zip_f      = self.aux.factors_order(self.F, self.K)
        # first column (FF5)
        self._FF5()
        # second column (HXZ)
        self._HXZ()
        # last columns (MOM)
        self._MOM()

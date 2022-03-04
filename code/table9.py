import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from aux_class import AuxClass

class Table9(object):

    def __init__(self, master_file, rf_file, lags):
        self.aux = AuxClass(master_file, rf_file)

        # self.df     = pd.read_excel(master_file, sheet_name="T1_ptfs", names=["dt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
        # self.df_rf  = pd.read_csv(rf_file, sep=";", names=["dtt", "v2", "v3", "v4", "rf"], dtype=str)
        #
        self.ann    = 12
        self.lags   = lags
        # Factors
        self.F      = self.df.keys().tolist()
        self.K      = len(self.F)
        self.zip_f  = {}
        self.df_rf  = self.df_rf[["dtt", "rf"]]
        self.df2    = self.df_rf.copy()
        self.T      = len(self.df)
from cmath import nan
from imp import init_builtin
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from aux_class import AuxClass

class Table9(object):

    def __init__(self, master_file, rf_file, num_firms):
        #self.aux = AuxClass(master_file, rf_file)
        self.num_firms = num_firms
        self.df = {}
        self.dates = {}
        for num in num_firms:
            header = [n for n in range(1, num+1)]
            self.df[num] = pd.read_csv("data/"+f"{num}"+"_industry_pfs.csv", sep=";", names= header)
            # fix date format.
            self.df[num].index = pd.to_datetime(self.df[num].index, format="%Y%m").strftime("%Y-%m")
            self.dates[num]    = self.df[num].index.tolist()
    
        # Estimations are done from 1972 to 2012
        self.init_year = 1972
        self.last_year   = 2021


        #self.master_df     = pd.read_excel(master_file, sheet_name="T1_ptfs", names=["dt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
        # self.df_rf  = pd.read_csv(rf_file, sep=";", names=["dtt", "v2", "v3", "v4", "rf"], dtype=str)

    def _df_set_time_frame(self):
        for num in self.num_firms:
            init_index  = self.dates[num].index(str(self.init_year) + "-01")
            end_index   = self.dates[num].index(str(self.last_year) + "-12") 
            self.df[num] = self.df[num][init_index:end_index]


    def _df_remove_invalid(self):
        df = self.df[30]
        N = len(df.columns)
        T = len(df.index)
        for n in range(1,N+1):
            mean = np.mean(df[n])
            for i in df.index.tolist():
                if df[n][i] == 0 or df[n][i] == nan:
                    df[n][i] = mean
    
    def _df_calculate_returns(self):
        self._df_remove_invalid()
        df = self.df[30]
        R = {}
        for n in df.columns:
            T = df[n].index.tolist()[:-2]
            print(T)
            for t in T:
                R[t+1] = df[n][t+1] / df[n][t]

        R = pd.DataFrame(R)
        print(R)
    def Table9(self):
        #self._df_calculate_returns()
        self._df_set_time_frame()
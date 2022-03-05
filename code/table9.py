from cmath import nan
from imp import init_builtin
from unicodedata import name
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from aux_class import AuxClass

class Table9(object):

    def __init__(self, master_file, rf_file, num_firms):
        #self.master_df     = pd.read_excel(master_file, sheet_name="T1_ptfs", names=["dt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
        # self.df_rf  = pd.read_csv(rf_file, sep=";", names=["dtt", "v2", "v3", "v4", "rf"], dtype=str)

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
        self.last_year   = 2012

    def _df_set_time_frame(self):
        for num in self.num_firms:
            init_index  = self.dates[num].index(str(self.init_year) + "-01")
            end_index   = self.dates[num].index(str(self.last_year) + "-12") 
            self.df[num] = self.df[num][init_index:end_index]

    def _df_remove_invalid_pfs(self):
        for num in self.num_firms:
            N = len(self.df[num].columns)
            T = len(self.df[num].index)
            for n in range(1,N+1):
                mean = np.mean(self.df[num][n])
                for i in self.df[num].index.tolist():
                    if self.df[num][n][i] == 0 or self.df[num][n][i] == nan:
                        self.df[num][n][i] = mean

    def _df_build_return_dataset(self):
        df = self.df[30]
        T = df.index.tolist()
        R = {'dt' : T[1:]}
        for n in df.columns:
            R[n] = []
            for t in T[:-1]:
                Rtt = df[n][T[T.index(t)+1]] / df[n][t]
                R[n].append(Rtt)
        R = pd.DataFrame(R).set_index('dt')
        R = R.rename_axis(index=None, columns=None)
        print(R.head())

    def Table9(self):
        # Take the correct time frame
        self._df_set_time_frame()
        # Remove every invalid pff (0 and NaN)
        self._df_remove_invalid_pfs()
        # Build Return data frame
        self._df_build_return_dataset()

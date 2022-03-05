from cmath import nan
from numpy.linalg import inv
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from aux_class import AuxClass

class Table9(object):

    def __init__(self, master_file, monthly_factors, num_firms):
        # Aux class with many functions
        self.aux = AuxClass(master_file, monthly_factors)

        # Read (monthly) factor file and put it fancy
        self.F = {}
        self.f = pd.read_csv(monthly_factors, sep=";", names=["dt", "mktrf", "smb", "hml", "rf"], dtype=str)
        self.f = self.aux.set_date_as_index(self.f[["dt", "mktrf", "smb", "hml"]])

        # Read (monthly) LL factor files
        self.LL = {}
        if 30 in num_firms:
            self.LL30           = pd.read_excel(master_file, sheet_name="T1_ptfs", names=["dt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
            self.LL30           = self.aux.set_date_as_index(self.LL30)
            self.LL30["LL"]     = pd.to_numeric(self.LL30["LL"], errors='coerce')*100
            self.LL[30]         = self.LL30[["LL"]]
        if 38 or 49 in num_firms:
            self.LL38and49      = pd.read_excel(master_file, sheet_name="T3_ptfs", names=["dt", "LL38", "LLStrong", "LL49", "LLStrong49"], dtype=str)
            self.LL38and49           = self.aux.set_date_as_index(self.LL38and49)
            if 38 in num_firms:
                self.LL38           = self.LL38and49[["LL38"]]
                self.LL38["LL38"]   = pd.to_numeric(self.LL38["LL38"], errors='coerce')*100
                self.LL[38]         = self.LL38
            if 49 in num_firms: 
                self.LL49           = self.LL38and49[["LL49"]]
                self.LL49["LL49"]   = pd.to_numeric(self. LL49["LL49"], errors='coerce')*100
                self.LL[49]         = self.LL49

        # List with the number of firms that we are using
        self.num_firms  = num_firms
        self.lags       = 12

        # Constructing every data set
        self.df = {}
        self.dates = {}
        for num in num_firms:
            header = [n for n in range(1, num+1)]
            self.df[num]        = pd.read_csv("data/"+f"{num}"+"_industry_pfs.csv", sep=";", names= header)
            # fix date format.
            self.df[num].index  = pd.to_datetime(self.df[num].index, format="%Y%m").strftime("%Y-%m")
            self.dates[num]     = self.df[num].index.tolist()
    
        # Estimations are done from 1972 to 2012
        self.init_year = 1972
        self.last_year = 2012

        # Returns
        self.R = {}
        # FR
        self.allFR = {}
        # b
        self.b = {}

    def _df_set_time_frame(self, df = None):
        if df is None: 
            for num in self.num_firms:
                init_index  = self.dates[num].index(str(self.init_year) + "-01")
                end_index   = self.dates[num].index(str(self.last_year) + "-12") 
                self.df[num] = self.df[num][init_index:end_index]
        else:
            init_index  = df.index.tolist().index(str(self.init_year) + "-01")
            end_index   = df.index.tolist().index(str(self.last_year) + "-12") 
            df = df[init_index:end_index]
            return df

    def _df_remove_invalid_pfs(self):
        for num in self.num_firms:
            N = len(self.df[num].columns)
            T = len(self.df[num].index)
            for n in range(1,N+1):
                mean = np.mean(self.df[num][n])
                for i in self.df[num].index.tolist():
                    if self.df[num][n][i] == 0 or self.df[num][n][i] == nan:
                        self.df[num][n][i] = mean

    def _df_build_return_datasets(self):
        for num in self.num_firms:
            df = self.df[num]
            T = df.index.tolist()
            self.R[num] = {'dt' : T[1:]}   
            for n in df.columns:
                self.R[num][n] = []
                for t in T[:-1]:
                    Rtt = df[n][T[T.index(t)+1]] / df[n][t]
                    self.R[num][n].append(Rtt)
            self.R[num] = pd.DataFrame(self.R[num]).set_index('dt')
            self.R[num] = self.R[num].rename_axis(index=None, columns=None)

    def _df_build_factors_dataset(self):
        # Take the correct time frame
        self.f = self._df_set_time_frame(self.f)
        for num in self.num_firms:
            # Join the three FF factors with the LL factor
            self.F[num] = self.f.join(self.LL[num])
        
    def _E_d(self):
        self.allFR = {}
        # Loop for each number of firms (we get a dataframe)
        for num in self.num_firms:
            # Rename the factor df
            F = self.F[num]
            # Remove the first row (because the returns are calculated only after t=1)
            F = F.iloc[1:,:]
            # Rename the return df
            R = self.R[num]
            # Create the dictionary for the products of F and R associated to the situation 
            # in which there are "num" firms
            regFR = {}
            # For a given number of firms, we loop among each one (we get temporal columns)
            for n in R.columns.tolist():
                # For EACH firm, create the FR dictionary with the products of factors and returns
                FR = {}
                # Loop among each factor, which are going to be in the header of a new data
                # frame containing the products FR
                for f in F.columns.tolist():
                    # Multiply through each date
                    FR[f] = F[f].mul(R[n])
                FR = pd.DataFrame(FR)
                FR = self.aux.include_constant_columns(FR)
                x = FR["const"]
                regs = []
                for f in FR.columns.tolist()[1:]:
                    y = FR[f]
                    reg = sm.OLS(y.astype(float), x.astype(float)).fit(cov_type='HAC', cov_kwds={'maxlags':self.lags})                    
                    regs.append(float(reg.params))
                # Include the regression in a dictionary (with the regressions for the 4 factors FOR EVERY FIRM)
                regFR[n] = regs
            regFR = pd.DataFrame(regFR)
            regFR.index = ["mktrf", "smb", "hml", "LL"]
            regFR = self.aux.replace_nan_by_row(regFR)
            # Include the resuls, as A MATRIX, to the dictionary with all the different situation (30, 38, 48)
            self.allFR[num] = regFR.values
        
    def _compute_b(self):
        for num in self.num_firms:
            E_d = self.allFR[num]
            self.b[num] = inv(E_d @ np.transpose(E_d)) @ E_d @ np.ones((num, 1))
            self.b[num] = [k[0] for k in self.b[num]]
        self.b = pd.DataFrame(self.b)
        self.b.index = ["mktrf", "smb", "hml", "LL"]
        print(self.b)

    def Table9(self):
        # Take the correct time frame
        self._df_set_time_frame()
        # Remove every invalid pff (0 and NaN)
        self._df_remove_invalid_pfs()
        # Build Return data frame
        self._df_build_return_datasets()
        # Take factors
        self._df_build_factors_dataset()
        # Calculate the "E[d] = E[FR]"
        self._E_d()
        # Compute the bs
        self._compute_b()

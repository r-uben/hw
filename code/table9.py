from numpy.linalg import inv
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.iolib.summary2 import summary_col
from aux_class import AuxClass

class Table9(object):

    def __init__(self, master_file, monthly_factors, num_firms):
        # Aux class with many functions
        self.aux = AuxClass(master_file, monthly_factors)

        # Files
        self.master_file = master_file
        self.monthly_factors = monthly_factors

        # Read (monthly) factor file and put it fancy
        self.F = {}
        self.f = pd.read_csv(monthly_factors, sep=";", names=["dt", "mktrf", "smb", "hml", "rf"], dtype=str)
        self.f = self.aux.set_date_as_index(self.f[["dt", "mktrf", "smb", "hml"]])

        # Read (monthly) LL factor files
        self.LL = {}

        # List with the number of firms that we are using
        self.num_firms  = num_firms
        self.lags       = 12

        # Factors
        self.factors = ["mktrf", "smb", "hml", "LL"]
        # Constructing every data set
        self.df = {}
        self.dates = {}
    
        # Estimations are done from 1972 to 2012
        self.init_year = 1972
        self.last_year = 2012

        # Returns
        self.R = {}
        # FR
        self.allFR = {}
        # b
        self.b = {}
        # eps
        self.S = {}
        # SE_bhat
        self.SE_b = {}
        # t_statistics
        self.t_stats = {}
        # aux
        self.Eff = {}
        # Lambda
        self.λ = {}

    def _read_files(self):
        if 30 in self.num_firms:
            LL30               = pd.read_excel(self.master_file, sheet_name="T1_ptfs", names=["dt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
            LL30               = self.aux.set_date_as_index(LL30)
            LL30.loc[:,"LL"]   = (pd.to_numeric(LL30["LL"], errors='coerce')*100).tolist()
            self.LL[30]        = LL30[["LL"]]
        if 38 or 49 in self.num_firms:
            LL38and49          = pd.read_excel(self.master_file, sheet_name="T3_ptfs", names=["dt", "LL38", "LLStrong", "LL49", "LLStrong49"], dtype=str)
            LL38and49          = self.aux.set_date_as_index(LL38and49)
            if 38 in self.num_firms:
                LL38                = LL38and49[["LL38"]]
                LL38.loc[:, "LL38"] = (pd.to_numeric(LL38["LL38"], errors='coerce')*100).tolist()
                self.LL[38]         = LL38
            if 49 in self.num_firms: 
                LL49                = LL38and49[["LL49"]]
                LL49.loc[:,"LL49"]  = (pd.to_numeric(LL49["LL49"], errors='coerce')*100).tolist()
                self.LL[49]         = LL49
    
    def _construct_df(self):
        for num in self.num_firms:
            header = [n for n in range(1, num+1)]
            self.df[num]        = pd.read_csv("data/"+f"{num}"+"_industry_pfs.csv", sep=";", names= header)
            # fix date format.
            self.df[num].index  = pd.to_datetime(self.df[num].index, format="%Y%m").strftime("%Y-%m")
            self.dates[num]     = self.df[num].index.tolist()

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

    def _df_build_return_datasets(self):
        for num in self.num_firms:
            df = self.df[num]
            T = df.index.tolist()
            self.R[num] = {'dt' : T}   
            for n in df.columns:
                self.R[num][n] = []
                for t in T:
                    Rt = df[n][t] + 1 #/ df[n][t]
                    self.R[num][n].append(Rt)
            self.R[num]  = pd.DataFrame(self.R[num]).set_index('dt')
            column_means = self.R[num].mean()
            self.R[num]  = self.R[num].fillna(0) 
            self.R[num]  = self.R[num].rename_axis(index=None, columns=None)

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
                for f in FR.columns.tolist():
                    y = FR[f]
                    reg = sm.OLS(y.astype(float), x.astype(float)).fit(cov_type='HAC', cov_kwds={'maxlags':self.lags})                    
                    regs.append(float(reg.params))
                # Include the regression in a dictionary (with the regressions for the 4 factors FOR EVERY FIRM)
                regFR[n] = regs[1:]
            regFR = pd.DataFrame(regFR)
            regFR.index = ["mktrf", "smb", "hml", "LL"]
            # Include the resuls, as A MATRIX, to the dictionary with all the different situation (30, 38, 48)
            regFR = self.aux.replace_nan_by_row(regFR)
            self.allFR[num] = regFR.values
        
    def _compute_b(self):
        '''
            Here we must compute the loadings of the SDF. To so so, we use the fact that
            b = inv(E[d]E[d]')E[d]-1, as seen in class.
        '''
        self._E_d() 
        for num in self.num_firms:
            E_d = self.allFR[num]
            self.b[num] = inv(E_d @ np.transpose(E_d)) @ E_d @ np.ones((num, 1))
            self.b[num] = [k[0] for k in self.b[num]]
        self.b = pd.DataFrame(self.b)
        self.b.index = ["mktrf", "smb", "hml", "LL"]
        print()
        print()
        print("############# TABLE 9 #############")
        print()
        print(self.b)

    def _compute_S(self):
        '''
            Here we must compute the errors in the GMM model. They are given by the formula
            ε_t^i = b F_t R_t^i - 1. Note that here b and F are vectors in R^K and R_t^i is a number.
            If we take the complete column, we have (1xK) x (KxT) x (Tx1) = (1xK) x (Kx1)= 1x1. 
            So that's what we are gonna do, we are gonna take the matrix FR that has been kept in self.allFR, 
            and multiply each column by the estimated parameter b. Then, substract one to it.  

            Finally, we compute S = V(ε) = 1/T (Dg' S^{-1} Dg)^{-1}, take its diagonal and then the square root.
        '''
        for num in self.num_firms:
            F     = self.F[num]
            R     = self.R[num]
            bhat  = self.b[num].values
            ε_num = {}
            for t in F.index.tolist():
                ε_num_t = []
                for n in R.columns.tolist():
                    ε_num_t.append(np.inner(bhat, F.loc[t].values) * R.loc[t,n] - 1)
                ε_num[t] = ε_num_t
            ε_num = pd.DataFrame(ε_num, columns=F.index.tolist())
            self.S[num] = np.cov(ε_num, dtype=float)

    def _compute_SE_b(self):
        for num in self.num_firms:
            T = len(self.F[num])
            Dg = self.allFR[num]
            VE_b = 1/T *  inv(Dg @ inv(self.S[num]) @ np.transpose(Dg)) 
            self.SE_b[num] = [np.sqrt(a) for a in VE_b.diagonal()]
        self.SE_b = pd.DataFrame(self.SE_b)
        self.SE_b.index = self.factors 
        print()
        print()
        print("########### STANDARD ERRORS ##########")
        print()
        print(self.SE_b)

    def _compute_t_stats(self):
        for num in self.num_firms:
            t_stat = []
            for f in self.b.index.tolist():
                bhat = self.b.loc[f, num]
                se   = self.SE_b.loc[f, num]
                t_stat.append(abs(bhat/se))
            self.t_stats[num] = t_stat
        self.t_stats = pd.DataFrame(self.t_stats)
        self.t_stats.index =  self.factors
        print()
        print()
        print("############## T STATS ##############")
        print()
        print(self.t_stats)

    def _compute_Eff(self):
        self.ff = {}
        for num in self.num_firms:
            F  = self.F[num]
            ff = []
            for t in F.index.tolist():
                Ft = F.loc[t,:].values
                print(np.inner(Ft,Ft))
                ff.append( np.inner(Ft,Ft) )
            self.ff[num] = ff
        self.ff = pd.DataFrame(self.ff)
        self.ff = self.aux.include_constant_columns(self.ff)
        x = self.ff["const"] 
        for num in self.num_firms:
            y = self.ff[num]
            reg = sm.OLS(y.astype(float), x.astype(float)).fit(cov_type='HAC', cov_kwds={'maxlags':self.lags})                    
            self.Eff[num] = float(reg.params)
        print(self.Eff)

    def _compute_λ(self):
        self._compute_Eff()
        for num in self.num_firms:
            self.λ[num] = self.b[num].values * self.Eff[num]
        self.λ = pd.DataFrame(self.λ)
        self.λ.index = self.factors
        print()
        print()
        print(self.λ)

    
    def Table9(self):
        self._read_files()
        self._construct_df()
        self._df_set_time_frame()
        self._df_build_return_datasets()
        self._df_build_factors_dataset()
        # Calculate the "E[d] = E[FR]"
        self._compute_b()
        self._compute_S()
        self._compute_SE_b()
        self._compute_t_stats()
        self._compute_λ()

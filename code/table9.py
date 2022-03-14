from numpy.linalg import inv

import math
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
        self.rf = self.aux.set_date_as_index(self.f[["dt", "rf"]]) 
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
        self.Re = {}
        self.ERe = {}
        # FR
        self.allFR = {}
        # b
        self.b = {}
        self.b2 = {}
        # eps
        self.S = {}
        # SE_bhat
        self.SE_b = {}
        # t_statistics
        self.t_stats = {}
        # aux
        self.allEff = {}
        # Lambd
        self.λ = {}

    def _read_files(self):
        if 30 in self.num_firms:
            LL30               = pd.read_excel(self.master_file, sheet_name="T1_ptfs", names=["dt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
            LL30               = self.aux.set_date_as_index(LL30)
            LL30.loc[:,"LL"]   = (pd.to_numeric(LL30.loc[:,"LL"], errors='coerce')*100).tolist() 
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
            init_index = 0
            end_index = 0
            for num in self.num_firms:
                init_index  = self.dates[num].index(str(self.init_year) + "-01")
                # +1 to include the last date
                end_index   = self.dates[num].index(str(self.last_year) + "-12")+1
                self.df[num] = self.df[num][init_index:end_index]
        else:
            init_index  = df.index.tolist().index(str(self.init_year) + "-01")
            # +1 to include the last date
            end_index   = df.index.tolist().index(str(self.last_year) + "-12")+1
            df = df[init_index:end_index]
            return df

    def _df_build_factors_dataset(self):
        # Take the correct time frame
        self.f = self._df_set_time_frame(self.f)
        self.rf = self._df_set_time_frame(self.rf)
        for num in self.num_firms:
            # Join the three FF factors with the LL factor
            LL = self.LL[num].columns[0]
            # excess
            #åself.LL[num].loc[:,LL] =  [self.LL[num].loc[t, LL] - self.rf.loc[t,'rf'] for t in self.LL[num].index]
            self.F[num] = self.f.join(self.LL[num]) 

    def _df_build_return_datasets(self):

        self._df_build_factors_dataset()

        for num in self.num_firms:
            #self.ERe[num] = []
            df = self.df[num]
            T = df.index.tolist()
            self.Re[num] = {'dt' : T}   
            for n in df.columns:
                self.Re[num][n] = []
                for t in T:
                    #Excess returns
                    Rt = df.loc[t, n] - self.rf.loc[t,'rf']#/ df[n][t]
                    self.Re[num][n].append(Rt)
                #self.ERe[num].append(np.mean(self.Re[num][n]))
            # Add LL
            self.Re[num][n+1] = self.F[num][self.F[num].columns[-1]] #- self.rf.loc[:, 'rf'] 
            #self.ERe[num].append(np.mean(self.Re[num][n+1]))
            # Replace NaN in ERe
            #self.ERe[num] = [0 if math.isnan(x) else x for x in self.ERe[num]]
            #self.ERe[num] = np.array(self.ERe[num])  
            self.Re[num]  = pd.DataFrame(self.Re[num]).set_index('dt')
            column_means = self.Re[num].mean()
            self.Re[num]  = self.Re[num].fillna(0) 
            self.Re[num]  = self.Re[num].rename_axis(index=None, columns=None)


    def _E_d(self):
        self.allFR = {}
        # Loop for each number of firms (we get a dataframe)
        for num in self.num_firms:
            # Rename the factor df
            F = self.F[num]
            # Rename the return df
            R = self.Re[num]
            R = R.fillna(0)
            self.ERe[num] = []
            # Create the dictionary for the products of F and R associated to the situation 
            # in which there are "num" firms
            regFR = {}
            # For a given number of firms, we loop among each one (we get temporal columns)
            for n in R.columns:
                self.ERe[num].append(np.mean(self.Re[num][n]))
                # For EACH firm, create the FR dictionary with the products of factors and returns
                FR = {}
                # Loop among each factor, which are going to be in the header of a new data
                # frame containing the products FR
                for f in F.columns:
                    # Multiply through each date
                    FR[f] = np.multiply(F[f], R[n])
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
            #  #= self.aux.replace_nan_by_row(regFR)
            self.ERe[num]  = np.array(self.ERe[num])
            self.allFR[num] = regFR.values
        
    def _compute_b(self):
        '''
            Here we must compute the loadings of the SDF. To so so, we use the fact that
            b = inv(E[d]E[d]')E[d]E[Re], as seen in class.
        '''
        self._E_d() 
        for num in self.num_firms:
            E_d = self.allFR[num]
            self.b[num] = inv(E_d @ np.transpose(E_d)) @ E_d @ np.transpose(self.ERe[num])#np.ones((len(self.Re[num].columns), 1))
        self.b = pd.DataFrame(self.b)
        self.b.index = ["mktrf", "smb", "hml", "LL"]
        print()
        print()
        print("############# TABLE 9 #############")
        print()
        print(self.b)
        self._compute_S()
        for num in self.num_firms:
            E_d = self.allFR[num]
            self.b2[num] = inv(E_d @ inv(self.S[num]) @ np.transpose(E_d)) @ E_d @ inv(self.S[num]) @ np.transpose(self.ERe[num])#np.ones((len(self.Re[num].columns), 1))
        self.b2 = pd.DataFrame(self.b2)
        self.b2.index = ["mktrf", "smb", "hml", "LL"]
        print()
        print()
        print("############# SECOND STAGE #############")
        print()
        print(self.b2)

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
            R     = self.Re[num]
            bhat  = self.b[num].values
            ε_num = {}
            for t in F.index.tolist():
                ε_num_t = []
                for n in R.columns.tolist():
                    ε_num_t.append(np.inner(bhat, F.loc[t].values) * R.loc[t,n])
                ε_num[t] = ε_num_t
            ε_num = pd.DataFrame(ε_num, columns=F.index.tolist())
            self.S[num] = np.cov(ε_num, dtype=float)

    def _compute_SE_b(self):
        for num in self.num_firms:
            T = len(self.F[num])
            Dg = self.allFR[num]
            VE_b = 1/(T-1) *  inv(Dg  @ np.transpose(Dg)) 
            # Add inv(self.S[num]) in the second step
            self.SE_b[num] = [np.sqrt(a)*10 for a in VE_b.diagonal()]
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
        self.allEff = {}
        for num in self.num_firms:
            F  = self.F[num]
            ff = {}
            for fcol in F.columns.tolist():
                ff[fcol] = []
                for frow in F.columns.tolist(): 
                    ff[fcol].append((np.cov(F[frow], F[fcol])[0][1] + np.mean(F[frow])*np.mean(F[fcol])))
                    # ff[frow] = np.multiply(F[fcol],F[frow])
            self.allEff[num] = pd.DataFrame(ff)
  

    def _compute_λ(self):
        self._compute_Eff()
        for num in self.num_firms:
            self.λ[num] = self.allEff[num] @ np.transpose(self.b[num].values)
        self.λ = pd.DataFrame(self.λ)
        self.λ.index = self.factors
        print()
        print()
        print("#################### λ ####################")
        print()
        print(self.λ)
        print()
        print()
        for num in self.num_firms:
            self.λ[num] = self.allEff[num].values @ np.transpose(self.b2[num].values)
        self.λ = pd.DataFrame(self.λ)
        self.λ.index = self.factors
        print()
        print("################# λ (2ND STAGE) #################")
        print()
        print(self.λ)
        print()
        print()

    
    def Table9(self):
        self._read_files()
        self._construct_df()
        self._df_set_time_frame()
        self._df_build_return_datasets()
        # Calculate the "E[d] = E[FR]"
        self._compute_b()
        self._compute_S()
        self._compute_SE_b()
        self._compute_t_stats()
        #self._compute_Eff()
        self._compute_λ()

from pickle import NONE
import pandas as pd
from pathlib import Path
class AuxClass(object):

    def __init__(self, master_file, rf_file):
        # DATA SETS
        self.df     = pd.read_excel(master_file, sheet_name="T3_ptfs", names=["dt", "LL38", "LLStrong38", "LL49", "LLStrong49"], dtype=str)
        self.df_rf  = pd.read_csv(rf_file, sep=";", names=["dtt", "v2", "v3", "v4", "rf"], dtype=str)
        self.df_rf  = self.df_rf[["dtt", "rf"]]
        self.df2    = self.df_rf.copy()
        self.df_ff  = pd.read_excel(master_file, sheet_name="T1_ptfs", names=["dttt", "LeadR", "MidR", "LagR", "Lead", "Mid", "Lag", "LL", "LLStrong", "mktrf", "smb", "hml"], dtype=str)
        self.df_ff  = self.df_ff[["dttt", "mktrf", "smb", "hml"]]
        self.df3    = self.df_ff.copy()

        self.zip_f  = {}

    def include_constant_columns(self, df):
        T = len(df)
        df["const"] = [1]*T
        cols    = df.columns.tolist()
        cols    = ["const"] + cols[:-1]
        df = df[cols]
        return df

    def set_date_as_index(self, df, cols = True):
        if "dt" in df.columns:
            df_new = df.copy()
            df_new = df_new.assign(dt = [x[:6] for x in df.dt])
            df_new = df_new.set_index("dt")
            df_new = df_new.rename_axis(index=None, columns=None)
            df_new.index = pd.to_datetime(df_new.index, format="%Y%m").strftime("%Y-%m")
            df_new = df_new.astype(float)
            return df_new
        else:
            return 'Please, add a temporal column'

    def remove_elements(self, L, to_remove):
        for l in to_remove:
            L.remove(l)
        return L

    def take_factors(self, df = None):
        if df is None:
            F = self.df.keys().tolist()
        else:
            F = df.keys().tolist()
        K = len(F)
        return F, K

    def factors_order(self, F, K):
        zip_f = {} 
        flags   = [k for k in range(K)]
        for flag in flags:
            zip_f[F[flag]] = flag
        return zip_f

    def _tex_file(self, table, title):
        return "results/subtables/" + table + "_" + title + ".txt"
        
    def create_txt(self, table, title, text):
        file    = Path(self._tex_file(table, title))
        file.write_text(f"{text}\n\n",)

    def replace_nan_by_column(self,df):
        # Thanks to Andy Hayden: https://stackoverflow.com/questions/33058590/pandas-dataframe-replacing-nan-with-row-average
        m = df.mean(axis=0)
        for col in df.columns:
            # using i allows for duplicate columns
            # inplace *may* not always work here, so IMO the next line is preferred
            # df.iloc[:, i].fillna(m, inplace=True)
            df[col] = df[col].fillna(m)
        return df
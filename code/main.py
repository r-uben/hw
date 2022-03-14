from table1 import Table1
from table3 import Table3
from table7 import Table7
from table9 import Table9

def find_file(name):
        return "data/" + name

if __name__=="__main__":

    ### MAIN DATASETS.

    master_file     = find_file("Data_master.xlsx")
    monthly_factors = find_file("FF4_monthlyKF.csv")

    ### TABLE 1 ###
    lags        = 24
    T1          = Table1(master_file, monthly_factors, lags)
    T1.Table1()

    ### TABLE 3 ###
    lags        = 12
    T3          = Table3(master_file, monthly_factors, lags)
    T3.Table3()

    ### TABLE 7 ###
    T7          = Table7(master_file, monthly_factors, lags)
    T7.Table7()

    ### TABLE 9 ###
    num_firms = [30, 38, 49, '25_book_size_all']
    T9          = Table9(master_file, monthly_factors, num_firms)
    T9.Table9()
from table1 import Table1
from table3 import Table3
from table7 import Table7

def find_file(name):
        return "data/" + name

if __name__=="__main__":

    master_file  = find_file("Data_master.xlsx")
    rf_file      = find_file("FF4_monthlyKF.csv")
    T1           = Table1(master_file, rf_file, 24)
    T1.Table1()
    T3           = Table3(master_file, rf_file, 12)
    T3.Table3()
    T7           = Table7(master_file, rf_file, 12)
    T7.Table7()
from table_one import Table1
from table_three import Table3

def find_file(name):
        return "data/" + name

if __name__=="__main__":

    master_file     = find_file("Data_master.xlsx")
    rf_file         = find_file("FF4_monthlyKF.csv")
#     files           = Table1(master_file, rf_file)
#     files.Table1()
    files           = Table3(master_file, rf_file)
    files.Table3()
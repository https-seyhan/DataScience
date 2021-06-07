import os
os.chdir('/home/saul/Business')
from openpyxl import Workbook
from openpyxl import load_workbook

def test():
    print("Test")
    workbook = Workbook()
    sheet = workbook.active
    sheet["A1"] = "hello"
    sheet["B1"] = "world!"
    workbook.save(filename="hello_world.xlsx")

def excel_work():
     workbook = load_workbook(filename="sample.xlsx")
     worksheet = workbook['Sheet2']
     #print("Type ", worksheet)
     #sheet = workbook.active
     value = worksheet["C2"].value
     row_count = worksheet.max_row
     column_count = worksheet.max_column
     print("Row count ", row_count)
     print("Column ", column_count)
     for row in worksheet.iter_rows(values_only=True):
         print(row)
     #for row in worksheet.rows:
     #print (

if __name__ == '__main__':
    #test()
    excel_work()

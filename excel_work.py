import os
os.chdir('/home/saul/Business')

from openpyxl import Workbook
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def test():
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
     #print(value)
     row_count = worksheet.max_row
     column_count = worksheet.max_column
     print("Row count ", row_count)
     print("Column ", column_count)
     #for row in worksheet.iter_rows(values_only=True):
     worksheet.cell(row=33, column=2).value = 'https://www.softwaretestinghelp.com/python-openpyxl-tutorial/'
     worksheet.cell(row=34, column=2).value = 'https://openpyxl.readthedocs.io/en/latest/pandas.html'
     worksheet.insert_rows(13)
     workbook.save("sample.xlsx")

     print("Saved")
     #for row in worksheet.row
 
if __name__ == '__main__':
    #test()
    excel_work()

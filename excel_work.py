import numpy as np
import pandas as pd
import re
import os
from openpyxl import Workbook
from openpyxl import load_workbook
from openpyxl.utils import FORMULAE
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter
os.chdir('/home/saul/Business')

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
     #print(value)
     row_count = worksheet.max_row
     column_count = worksheet.max_column
     print("Row count ", row_count)
     print("Column ", column_count)
     #for row in worksheet.iter_rows(values_only=True):
         #print(row)
     worksheet.cell(row=33, column=2).value = 'https://www.softwaretestinghelp.com/python-openpyxl-tutorial/'
     worksheet.cell(row=34, column=2).value = 'https://openpyxl.readthedocs.io/en/latest/pandas.html'
     worksheet.insert_rows(13)
     workbook.save("sample.xlsx")
     print("Saved")
     #for row in worksheet.rows:
     #     print (row)

def get_function_ranges(cellFormula):
    print("Cell formula ", cellFormula)
    formula = cellFormula.translate({ord(char): None for char in '='})
    funct = formula.split("(")[0]
    print("Function ", funct)
    ranges = formula[formula.find("(")+1:formula.find(")")]
    print("Ranges ", ranges)
    start = ranges.split(':')[0]
    end = ranges.split(':')[1]
    return funct, start, end

def copy_formula():
    print("Copy Formula Called")
    workbook = load_workbook(filename="sample_v2.xlsx")
    worksheet = workbook['Sheet2']
    funct, start, end = get_function_ranges(worksheet["C3"].value)
    print("Function {} start at {} and ends at {}".format(funct, start, end))
    start_row = re.findall("\d+", start)[0]
    print("Start Row ", start_row)
    end_row = re.findall("\d+", end)[0]
    print("End Row ", end_row)
    shift_range = int(end_row) - int(start_row)
   
    #print("Type ", worksheet)
    #sheet = workbook.active
    value = worksheet["C2"].value
    print(value)
    row_count = worksheet.max_row
    column_count = worksheet.max_column

    print("Row count ", row_count)
    print("Column ", column_count)

    worksheet.insert_rows(3)
    row_shift = 3
    
  
    #new_range_start = 
    #new_range_end = 
         
     #Add formula 
    worksheet["D7"] = "=AVERAGE(D9:D26)"
    workbook.save("sample_v3.xlsx")


if __name__ == '__main__':
    #test()
    #excel_work()
    copy_formula()

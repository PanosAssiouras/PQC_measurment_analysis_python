import decimal
import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import fsolve
import pathlib
import seaborn as sns
from astropy import stats
from astropy.visualization import hist
from decimal import *


def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width.
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively.


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin.
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return(result)

# Denoting fitting function
def fit_func(x,a, b):
    return a * x + b

def integers_round_up(number,decimals=0):
    number_of_digits=0
    if type(number)==int:
        while number>10:
            number=number/10
            number_of_digits+=1
    number=round_up(number,decimals)
    number=number*10**number_of_digits
    print(number)

def integers_round_down(number,decimals=0):
    number_of_digits=0
    if type(number)==int:
        while number>10:
            number=number/10
            number_of_digits+=1
    number=round_down(number,decimals)
    number=number*10**number_of_digits
    print(number)

def number_of_digits_to_round(number):
    number=abs(number)
    numberOfdigits = 0
    if number > 10:
        numberOfdigits = 0
        while (number > 10):
            number = number /10
            numberOfdigits -= 1
        print(number)
    elif number<1.0:
        numberOfdigits = 0
        while (number < 1.0):
            number = number * 10
            numberOfdigits += 1
        print(number)
    else:
        numberOfdigits+=1
    return numberOfdigits


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


#def findIntersection(fun1,fun2,x0):
#     return fsolve(lambda x :fun1(x) - fun2(x),x0)


root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')




#sns.set()
#plt.style.use('seaborn')
##Loop in each file
for file_name in file_names:
    pos=file_name.find(".csv")
    #pos=file_name.find(".csv")
    if (pos!=-1):
        ##read data
        data=pd.read_csv(file_name, sep=",",skiprows=0)
        data_top = data.head()
        # iterating the columns
        col_names=[]
        for col in data.columns:
            col_names.append(col)
        print(col_names)

        #print("\nNumber of columns:")
        #print(len(data.columns))
        #print(data)
        #for column in range[1:len(data.columns)]:
        #    values=data.iloc[:column]

        fig=[]
        #fig, ax = plt.subplots()
        #fig.suptitle('CV MOS : first derivative')
        #ax.set_xlabel('Voltage [V]')
        #ax.set_ylabel('diff(Capacitance) [F]')
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        # gives a tuple of column name and series
        # for each column in the dataframe
        index=0
        #print("integers round up",integers_round_up(50681))
        #print("integers round down", integers_round_down(50681))
        #print("number_of_digits",number_of_digits_to_round(5267545))
        measurment=(file_name.split('/')[-1]).split('.')[0]
        print(os.path.basename(__file__))
        for (columnName, columnData) in data.iteritems():
            # print('Colunm Name : ', columnName)
            # print(type(columnData.values[0]))
            if type(columnData.values[0])==str:
                print("{} It is string".format(columnName))
            elif isinstance(columnData.values[0],np.float64):
                #NBR_BINS = freedman_diaconis(columnData.values, returnas="bins")
                number_of_bins=stats.freedman_bin_width(columnData.values, return_bins=False)
                print("number of bins= {}".format(number_of_bins))
                print("{} It is integer".format(columnName))
                digits=number_of_digits_to_round(columnData.values[0])
                fig = plt.figure(index)
                ax = fig.add_subplot()
                ax.set_xlabel(col_names[index], fontsize=15)
                ax.set_ylabel("Number of test-structures", fontsize=15)
                ax.set_title("Histogram of {} from {} measurment".format(col_names[index].split(' ')[0], measurment))
                #print('Column Contents : ', columnData.values)
                ax.hist(columnData.values,range=[(round_down(min(columnData.values),digits+1)-
                                0.5* abs(round_down(min(columnData.values),digits+1))),
                               round_up(max(columnData.values),digits+1)+
                               0.5* abs(round_up(max(columnData.values),digits+1))],
                        bins=20, alpha=0.5, color='blue', edgecolor='white', linewidth=1.2)
                fig.set_size_inches(width / 100, height / 100)
                fig.savefig("Histo"+"_"+columnName+"_"+measurment+".png", dpi=100)
            index+=1

        plt.show()


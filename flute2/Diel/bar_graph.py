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

# Denoting fitting function
def fit_func(x,a, b):
    return a * x + b

#def findIntersection(fun1,fun2,x0):
#     return fsolve(lambda x :fun1(x) - fun2(x),x0)

##Filechooser
fig2, ax2 = plt.subplots()


root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
file_names = filedialog.askopenfilenames(parent=root,title='Choose a file')
##Loop in each file
for file_name in file_names:
    pos=file_name.find(".csv")
    #pos=file_name.find(".csv")
    if (pos!=-1):
        ##read data
        data=pd.read_csv(file_name, sep=",",skiprows=1)
        print(data)
        file_name = os.path.splitext(file_name)[0]
        names = data.iloc[:,0]
        value = data.iloc[:,1]
        labels=[]
        serialnumber=[]
        orientation=[]
        type=[]
        for index, row in data.iterrows():
            name=row[0]
            #print(name)
            split=name.split("_")
            serialnumber.append(split[4])
            orientation.append(split[5])
            type.append(split[-1])
            print("orientation=",orientation)
            print("orientation=", serialnumber)
            labels.append(split[4]+" "+split[5])
            fig2, ax2.bar(split[4]+" "+split[5]+split[-1], row[1], 0.8,
                          alpha=0.5, ecolor='black', capsize=10, label='rotated')
            ax2.set_ylim(3.0E-12,4.0E-12)
            ax2.set_ylabel('V_th [\u03A9/square]')
            ax2.set_title('FET measuremnts ')


        print(data)


        #ax2.set_xticks(x)
        #ax2.set_xticklabels(labels)
            #ax2.legend()


plt.show()

